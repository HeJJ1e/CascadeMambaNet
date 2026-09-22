from __future__ import annotations
import argparse
import copy
import json
import math
import os
import pickle
import random
import time
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

def _configure_cuda_visibility_from_cli() -> None:
    early_parser = argparse.ArgumentParser(add_help=False)
    early_parser.add_argument('--gpu', type=int, default=0)
    (early_args, _) = early_parser.parse_known_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(early_args.gpu)
_configure_cuda_visibility_from_cli()
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.amp import autocast
from torch.utils.data import DataLoader, IterableDataset
from configs.settings import ZAHAConfig as cfg
from datasets.zaha.metadata import CLASS_NAMES, NUM_CLASSES, SPLITS
from models.cascade_mamba_net import CascadeMambaNet
from models.losses import lovasz_softmax
from utils.ply import read_ply
from utils.pointcloud import DataProcessing as DP

def setup_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def log_out(message: str, log_file) -> None:
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

def amp_context():
    return autocast('cuda', dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()

class ModelEMA:

    def __init__(self, model, decay: float=0.999):
        self.ema = copy.deepcopy(model).eval()
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model) -> None:
        for (ema_parameter, parameter) in zip(self.ema.parameters(), model.parameters()):
            ema_parameter.data.mul_(self.decay).add_(parameter.detach().data, alpha=1.0 - self.decay)
        for (ema_buffer, buffer) in zip(self.ema.buffers(), model.buffers()):
            ema_buffer.data.copy_(buffer.data)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.ema.load_state_dict(state_dict)

def cbl_loss(logits, labels, neigh_idx, ignore_idx: int, temperature: float, num_classes: int):
    (batch_size, n_points) = labels.shape
    k_neighbors = neigh_idx.shape[-1]
    logits_flat = logits.reshape(batch_size * n_points, num_classes).float()
    labels_flat = labels.reshape(batch_size * n_points).long()
    batch_offset = (torch.arange(batch_size, device=labels.device) * n_points).view(batch_size, 1, 1)
    neigh_global = (neigh_idx + batch_offset).reshape(batch_size * n_points, k_neighbors).long()
    features = F.normalize(logits_flat, dim=-1)
    neigh_features = features[neigh_global]
    neigh_labels = labels_flat[neigh_global]
    valid_self = labels_flat != ignore_idx
    valid_neigh = neigh_labels != ignore_idx
    positive = (neigh_labels == labels_flat.unsqueeze(-1)) & valid_neigh
    boundary = positive.any(dim=-1) & (~positive & valid_neigh).any(dim=-1) & valid_self
    if not boundary.any():
        return logits_flat.sum() * 0.0
    similarity = (features.unsqueeze(1) * neigh_features).sum(dim=-1)[boundary]
    positive = positive[boundary].float()
    valid_neigh = valid_neigh[boundary].float()
    similarity = similarity / temperature
    similarity = similarity - similarity.max(dim=-1, keepdim=True)[0]
    exp_similarity = torch.exp(similarity) * valid_neigh
    positive_mass = (exp_similarity * positive).sum(dim=-1)
    all_mass = exp_similarity.sum(dim=-1)
    return (-torch.log((positive_mass + 1e-12) / (all_mass + 1e-12))).mean()

def cascade_collate_fn(batch, device=None):
    batch_xyz = np.stack([item[0] for item in batch])
    batch_labels = np.stack([item[1] for item in batch])
    batch_point_idx = np.stack([item[2] for item in batch])
    batch_cloud_idx = np.stack([item[3] for item in batch])
    batch_normals = np.stack([item[4] for item in batch])
    batch_geometries = np.stack([item[5] for item in batch])
    (batch_size, n_points, _) = batch_xyz.shape
    if n_points != cfg.stage_points[-1]:
        raise ValueError(f'batch contains {n_points} points but ZAHAConfig.stage_points[-1] is {cfg.stage_points[-1]}')
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = {}
    stage_xyz = []
    strides = [16, 8, 2, 1]
    for (stage_idx, stride) in enumerate(strides):
        selector = slice(None) if stride == 1 else np.arange(0, n_points, stride)
        xyz = batch_xyz[:, selector, :]
        labels = batch_labels[:, selector]
        normals = batch_normals[:, selector, :]
        geometries = batch_geometries[:, selector, :]
        feature_parts = [xyz, normals, geometries]
        features = np.concatenate(feature_parts, axis=-1).astype(np.float32, copy=False)
        expected_dim = 3 + 3 + cfg.geo_feature_dim
        if features.shape[-1] != expected_dim:
            raise RuntimeError(f'ZAHA feature dimension {features.shape[-1]} != expected {expected_dim}')
        neighbors = DP.knn_search(xyz, xyz, cfg.k_n)
        stage_xyz.append(xyz)
        inputs[f'xyz_s{stage_idx}'] = torch.from_numpy(xyz).float().to(device)
        inputs[f'neigh_idx_s{stage_idx}'] = torch.from_numpy(neighbors).long().to(device)
        inputs[f'features_s{stage_idx}'] = torch.from_numpy(features).float().to(device)
        inputs[f'labels_s{stage_idx}'] = torch.from_numpy(labels).long().to(device)
    for stage_idx in range(1, 4):
        upsample_idx = DP.knn_search(stage_xyz[stage_idx - 1], stage_xyz[stage_idx], k=3)
        inputs[f'upsample_idx_s{stage_idx}'] = torch.from_numpy(upsample_idx).long().to(device)
    inputs['labels'] = torch.from_numpy(batch_labels).long().to(device)
    inputs['input_inds'] = torch.from_numpy(batch_point_idx).long().to(device)
    inputs['cloud_inds'] = torch.from_numpy(batch_cloud_idx).long().to(device)
    return inputs

class PointCloudIterableDataset(IterableDataset):

    def __init__(self, generator_factory, samples_per_epoch: int):
        self.generator_factory = generator_factory
        self.samples_per_epoch = samples_per_epoch

    def __iter__(self):
        generator = self.generator_factory()
        for _ in range(self.samples_per_epoch):
            yield next(generator)

class ZAHA:

    def __init__(self, processed_data_root: str | Path | None=None):
        self.name = 'ZAHA_LoFG3'
        self.label_to_names = {index: name for (index, name) in enumerate(CLASS_NAMES)}
        self.label_values = np.arange(NUM_CLASSES, dtype=np.int32)
        self.label_to_idx = {int(label): int(label) for label in self.label_values}
        self.ignored_labels = np.array([], dtype=np.int32)
        self.processed_root = Path(processed_data_root or cfg.processed_data_root).expanduser().resolve()
        manifest_path = self.processed_root / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f'ZAHA manifest not found: {manifest_path}. Run python -m datasets.zaha.preprocess first.')
        with manifest_path.open('r', encoding='utf-8') as handle:
            self.manifest = json.load(handle)
        if self.manifest.get('num_classes') != NUM_CLASSES:
            raise ValueError('manifest is not a 15-class ZAHA LoFG3 preparation')
        if abs(float(self.manifest.get('grid_size', -1)) - float(cfg.sub_grid_size)) > 1e-09:
            raise ValueError(f"manifest grid {self.manifest.get('grid_size')} does not match ZAHAConfig.sub_grid_size={cfg.sub_grid_size}")
        if self.manifest.get('sampling_unit') != 'source_level_global_voxel_cloud':
            raise ValueError('ZAHA manifest uses an obsolete tile-level sampling layout; re-run python -m datasets.zaha.preprocess with the current code.')
        self.records = {split: list(self.manifest['splits'].get(split, [])) for split in SPLITS}
        if not self.records['train'] or not self.records['val'] or (not self.records['test']):
            raise ValueError('manifest must contain non-empty official train, val and test splits')
        self.train_class_counts = np.asarray(self.manifest['subsampled_class_counts']['train'], dtype=np.int64)
        if self.train_class_counts.shape != (NUM_CLASSES,) or np.any(self.train_class_counts <= 0):
            raise ValueError('manifest training class counts must cover all 15 ZAHA classes')
        self._cache: OrderedDict[tuple[str, int], dict] = OrderedDict()
        self._cache_capacity = max(int(cfg.max_cached_clouds), 1)
        self.min_possibility = {split: np.empty(0, dtype=np.float64) for split in SPLITS}
        self._sampling_initialized = {split: np.empty(0, dtype=bool) for split in SPLITS}
        self._sampling_state_dir: Path | None = None

    def _record_path(self, record: dict, key: str) -> Path:
        try:
            return self.processed_root / record[key]
        except KeyError as exc:
            raise KeyError(f"manifest record {record.get('id')} has no {key!r} path") from exc

    def _prepare_sampling(self, split: str) -> None:
        if split not in SPLITS:
            raise ValueError(f'unknown ZAHA split {split!r}')
        if self._sampling_state_dir is None:
            run_id = f'sampling_{os.getpid()}_{time.time_ns()}'
            self._sampling_state_dir = Path(cfg.sampling_state_root) / run_id
            self._sampling_state_dir.mkdir(parents=True, exist_ok=False)
        self.min_possibility[split] = np.zeros(len(self.records[split]), dtype=np.float64)
        self._sampling_initialized[split] = np.zeros(len(self.records[split]), dtype=bool)
        for key in [key for key in self._cache if key[0] == split]:
            payload = self._cache.pop(key)
            payload['possibility'].flush()

    def _possibility_path(self, split: str, record: dict) -> Path:
        assert self._sampling_state_dir is not None
        return self._sampling_state_dir / f"{split}__{record['id']}.possibility.f32"

    def _load_cloud(self, split: str, cloud_index: int) -> dict:
        key = (split, cloud_index)
        if key in self._cache:
            payload = self._cache.pop(key)
            self._cache[key] = payload
            return payload
        record = self.records[split][cloud_index]
        data = read_ply(str(self._record_path(record, 'ply')))
        fields = set(data.dtype.names or ())
        required = {'x', 'y', 'z', 'class'}
        if (missing := (required - fields)):
            raise ValueError(f"{record['id']} is missing processed fields {sorted(missing)}")
        forbidden = {'red', 'green', 'blue'} & fields
        if forbidden:
            raise ValueError(f"{record['id']} contains forbidden pseudo-RGB fields {sorted(forbidden)}")
        points = np.column_stack((data['x'], data['y'], data['z'])).astype(np.float32, copy=False)
        labels = data['class'].astype(np.int64, copy=False)
        if len(points) != int(record['sub_points']):
            raise ValueError(f"{record['id']} point count differs from manifest")
        if labels.size and (labels.min() < 0 or labels.max() >= NUM_CLASSES):
            raise ValueError(f"{record['id']} has labels outside [0, {NUM_CLASSES - 1}]")
        with self._record_path(record, 'tree').open('rb') as handle:
            tree = pickle.load(handle)
        normals = None
        geometry = None
        normals = np.load(self._record_path(record, 'normal'), mmap_mode='r')
        if normals.shape != (len(points), 3):
            raise ValueError(f"{record['id']} normal cache has wrong shape {normals.shape}")
        geometry = np.load(self._record_path(record, 'geo'), mmap_mode='r')
        if geometry.shape != (len(points), cfg.geo_feature_dim):
            raise ValueError(f"{record['id']} geometry cache has wrong shape {geometry.shape}")
        possibility_path = self._possibility_path(split, record)
        mode = 'r+' if possibility_path.exists() else 'w+'
        possibility = np.memmap(possibility_path, dtype=np.float32, mode=mode, shape=(len(points),))
        if not self._sampling_initialized[split][cloud_index]:
            possibility[:] = np.random.rand(len(points)).astype(np.float32) * 0.001
            possibility.flush()
            self._sampling_initialized[split][cloud_index] = True
            self.min_possibility[split][cloud_index] = float(np.min(possibility))
        payload = {'points': points, 'labels': labels, 'tree': tree, 'normals': normals, 'geometry': geometry, 'possibility': possibility}
        while len(self._cache) >= self._cache_capacity:
            (_, evicted) = self._cache.popitem(last=False)
            evicted['possibility'].flush()
        self._cache[key] = payload
        return payload

    def iter_projection_chunks(self, split: str, cloud_index: int):
        record = self.records[split][cloud_index]
        projection_paths = record.get('projection_chunks')
        if not isinstance(projection_paths, list) or not projection_paths:
            raise ValueError(f"split {split!r} has no raw-point projections for {record['id']}")
        for relative_path in projection_paths:
            projection_path = self.processed_root / relative_path
            with projection_path.open('rb') as handle:
                (proj_idx, labels) = pickle.load(handle)
            proj_idx = np.asarray(proj_idx, dtype=np.int32)
            labels = np.asarray(labels, dtype=np.int64)
            if len(proj_idx) != len(labels):
                raise ValueError(f'projection/label length mismatch for {projection_path}')
            yield (Path(relative_path).stem.replace('_proj', ''), proj_idx, labels)

    def get_batch_gen(self, split: str):
        self._prepare_sampling(split)
        use_mix3d = split == 'train'

        def sample_one():
            cloud_index = int(np.argmin(self.min_possibility[split]))
            cloud = self._load_cloud(split, cloud_index)
            points = cloud['points']
            possibility = cloud['possibility']
            point_index = int(np.argmin(possibility))
            center = points[point_index:point_index + 1]
            center = center + np.random.normal(scale=cfg.noise_init / 10.0, size=center.shape).astype(np.float32)
            query_size = min(len(points), cfg.num_points)
            queried_idx = cloud['tree'].query(center, k=query_size, return_distance=False).reshape(-1)
            queried_idx = DP.shuffle_idx(queried_idx)
            local_xyz = points[queried_idx] - center
            queried_labels = cloud['labels'][queried_idx]
            queried_normals = np.asarray(cloud['normals'][queried_idx], dtype=np.float32)
            queried_geometry = np.asarray(cloud['geometry'][queried_idx], dtype=np.float32)
            distances = np.sum(np.square(points[queried_idx] - center), axis=1)
            max_distance = max(float(distances.max()), 1e-12)
            possibility[queried_idx] += np.square(1.0 - distances / max_distance).astype(np.float32)
            self.min_possibility[split][cloud_index] = float(np.min(possibility))
            if query_size < cfg.num_points:
                duplicate = np.random.choice(query_size, cfg.num_points - query_size)
                pad_idx = np.concatenate((np.arange(query_size), duplicate))
                local_xyz = local_xyz[pad_idx]
                queried_labels = queried_labels[pad_idx]
                queried_idx = queried_idx[pad_idx]
                queried_normals = queried_normals[pad_idx]
                queried_geometry = queried_geometry[pad_idx]
            if split == 'train':
                theta = np.random.uniform(0.0, 2.0 * np.pi)
                rotation = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
                local_xyz = local_xyz @ rotation.T
                queried_normals = queried_normals @ rotation.T
                local_xyz *= np.random.uniform(0.9, 1.1)
                local_xyz += np.random.normal(0.0, 0.01, local_xyz.shape).astype(np.float32)
            return (local_xyz.astype(np.float32, copy=False), queried_labels.astype(np.int64, copy=False), queried_idx.astype(np.int32, copy=False), np.array([cloud_index], dtype=np.int32), queried_normals.astype(np.float32, copy=False), queried_geometry.astype(np.float32, copy=False))

        def generator():
            while True:
                if use_mix3d and np.random.random() < cfg.mix3d_prob:
                    (xyz_a, labels_a, idx_a, cloud_a, normals_a, geo_a) = sample_one()
                    (xyz_b, labels_b, _, _, normals_b, geo_b) = sample_one()
                    xyz_b = xyz_b.copy()
                    xyz_b[:, 0] += float(xyz_a[:, 0].max() - xyz_b[:, 0].min()) + cfg.mix3d_gap
                    combined_xyz = np.concatenate((xyz_a, xyz_b), axis=0)
                    combined_labels = np.concatenate((labels_a, labels_b), axis=0)
                    combined_normals = np.concatenate((normals_a, normals_b), axis=0)
                    combined_geo = np.concatenate((geo_a, geo_b), axis=0)
                    selected = np.random.choice(len(combined_xyz), cfg.num_points, replace=False)
                    yield (combined_xyz[selected], combined_labels[selected], idx_a, cloud_a, combined_normals[selected], combined_geo[selected])
                else:
                    yield sample_one()
        return generator

def evaluate(model, val_loader, log_file) -> float:
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for (step_index, batch_data) in enumerate(val_loader):
            if step_index % 50 == 0:
                print(f'validation batch {step_index}/{cfg.val_steps}')
            with amp_context():
                logits = model(batch_data)
            predictions = torch.argmax(logits, dim=-1).cpu().numpy().reshape(-1)
            labels = batch_data['labels'].cpu().numpy().reshape(-1)
            confusion += confusion_matrix(labels, predictions, labels=np.arange(NUM_CLASSES))
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = support + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=union > 0)
    class_acc = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    oa = true_positive.sum() / max(support.sum(), 1.0)
    miou = float(iou.mean() * 100.0)
    macc = float(class_acc.mean() * 100.0)
    log_out(f'sampled val OA={oa * 100:.2f} mIoU={miou:.2f} mAcc={macc:.2f}', log_file)
    log_out('sampled val IoU: ' + ' '.join((f'{name}={score * 100:.2f}' for (name, score) in zip(CLASS_NAMES, iou))), log_file)
    return miou

def train_network(model, dataset: ZAHA, resume: str | None=None) -> Path:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if resume:
        saving_path = Path(resume).expanduser().resolve()
        run_name = saving_path.name
    else:
        run_name = f"{cfg.experiment_name}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())}"
        saving_path = Path(cfg.results_root) / run_name
    log_path = Path(cfg.logs_root) / f'{run_name}.log'
    saving_path.mkdir(parents=True, exist_ok=True)
    Path(cfg.logs_root).mkdir(parents=True, exist_ok=True)
    (saving_path / 'snapshots').mkdir(exist_ok=True)
    cfg.saving_path = saving_path
    run_metadata = {'experiment': cfg.experiment_name, 'num_classes': NUM_CLASSES, 'class_names': list(CLASS_NAMES), 'use_color': cfg.use_color, 'processed_root': str(dataset.processed_root), 'preprocessing': {'label_protocol': dataset.manifest['label_protocol'], 'sampling_unit': dataset.manifest['sampling_unit'], 'grid_size_m': dataset.manifest['grid_size'], 'temporary_tile_voxels': dataset.manifest['tile_voxels'], 'normal_k': dataset.manifest['normal_k'], 'rgb_policy': dataset.manifest['rgb_policy']}, 'official_split_source_counts': dataset.manifest['split_source_counts'], 'evaluation_protocol': {'reported_split': 'official_test', 'full_raw_point_reprojection': True, 'tta': False}, 'optimization': {'learning_rate': cfg.learning_rate, 'weight_decay': cfg.weight_decay, 'max_epoch': cfg.max_epoch, 'warmup_epochs': cfg.warmup_epochs, 'eta_min': cfg.eta_min, 'ema_decay': cfg.ema_decay, 'lovasz_weight': cfg.lovasz_weight, 'cbl_weight': cfg.cbl_weight, 'boundary_weight': cfg.boundary_weight}, 'class_counts_train_subsampled': dataset.train_class_counts.tolist()}
    with (saving_path / 'run_metadata.json').open('w', encoding='utf-8') as handle:
        json.dump(run_metadata, handle, indent=2)
    train_dataset = PointCloudIterableDataset(dataset.get_batch_gen('train'), cfg.train_steps * cfg.batch_size)
    val_dataset = PointCloudIterableDataset(dataset.get_batch_gen('val'), cfg.val_steps * cfg.val_batch_size)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=cascade_collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.val_batch_size, collate_fn=cascade_collate_fn, drop_last=True)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.0005)

    def learning_rate_lambda(epoch: int) -> float:
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.max_epoch - cfg.warmup_epochs)
        minimum = cfg.eta_min / cfg.learning_rate
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_lambda)
    ema_model = ModelEMA(model, decay=cfg.ema_decay)
    class_weights = DP.get_zaha_class_weights(dataset.train_class_counts, cfg.class_weight_smoothing, cfg.class_weight_max)
    class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    checkpoint_path = saving_path / 'last_ckpt.pth'
    history = [0.0]
    start_epoch = 0
    global_step = 1
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if ema_model is not None and 'ema_state_dict' in checkpoint:
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
        history = checkpoint['mIou_list']
        start_epoch = int(checkpoint['epoch']) + 1
        global_step = int(checkpoint['global_step'])
    with log_path.open('a', encoding='utf-8') as log_file:
        log_out(f'ZAHA run={run_name}', log_file)
        log_out(f'weights={saving_path}', log_file)
        log_out(f'log={log_path}', log_file)
        log_out(f'class_weights={class_weights.tolist()}', log_file)
        log_out(f'loss=focal+lovasz lovasz_weight={cfg.lovasz_weight:.3f} cbl_weight={cfg.cbl_weight:.3f} boundary_weight={cfg.boundary_weight:.3f}', log_file)
        log_out(f'periodic_checkpoints=epoch{cfg.periodic_checkpoint_start_epoch} every {cfg.periodic_checkpoint_interval} epochs', log_file)
        if resume:
            log_out(f'resumed at epoch={start_epoch} step={global_step}', log_file)
        for epoch in range(start_epoch, cfg.max_epoch):
            log_out(f'**** EPOCH {epoch} ****', log_file)
            model.train()
            for batch_data in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with amp_context():
                    model_output = model(batch_data)
                predictions_by_stage = model_output
                total_loss = 0.0
                for (stage_idx, prediction) in enumerate(predictions_by_stage):
                    labels = batch_data[f'labels_s{stage_idx}'].reshape(-1)
                    logits = prediction.float().reshape(-1, NUM_CLASSES)
                    ce = F.cross_entropy(logits, labels, reduction='none')
                    focal = (1.0 - torch.exp(-ce)).pow(2.0)
                    sample_weights = class_weights_tensor[labels]
                    focal_loss = (focal * sample_weights * ce).mean()
                    lovasz = lovasz_softmax(logits, labels, classes='present')
                    stage_loss = (1.0 - cfg.lovasz_weight) * focal_loss + cfg.lovasz_weight * lovasz
                    stage_labels = batch_data[f'labels_s{stage_idx}']
                    neighbors = batch_data[f'neigh_idx_s{stage_idx}']
                    (batch_size, point_count) = stage_labels.shape
                    neighbor_labels = stage_labels.gather(1, neighbors.reshape(batch_size, -1)).reshape_as(neighbors)
                    boundary = (neighbor_labels != stage_labels.unsqueeze(-1)).any(dim=-1).reshape(-1)
                    weighted_ce = (focal * sample_weights * ce * (1.0 + (cfg.boundary_weight - 1.0) * boundary)).mean()
                    stage_loss = (1.0 - cfg.lovasz_weight) * weighted_ce + cfg.lovasz_weight * lovasz
                    total_loss = total_loss + cfg.stage_weights[stage_idx] * stage_loss
                    if stage_idx in cfg.cbl_stages:
                        total_loss = total_loss + cfg.cbl_weight * cbl_loss(prediction, batch_data[f'labels_s{stage_idx}'], batch_data[f'neigh_idx_s{stage_idx}'], -100, cfg.cbl_temperature, NUM_CLASSES)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                if ema_model is not None:
                    ema_model.update(model)
                if global_step % 50 == 0:
                    with torch.no_grad():
                        accuracy = (predictions_by_stage[-1].argmax(dim=-1) == batch_data['labels']).float().mean().item()
                    log_out(f'step={global_step:08d} loss={total_loss.item():.4f} acc={accuracy:.4f}', log_file)
                global_step += 1
            evaluation_model = ema_model.ema if ema_model is not None else model
            miou = evaluate(evaluation_model, val_loader, log_file)
            is_best = miou > max(history)
            is_periodic = epoch >= cfg.periodic_checkpoint_start_epoch and (epoch - cfg.periodic_checkpoint_start_epoch) % cfg.periodic_checkpoint_interval == 0
            if is_best or is_periodic:
                snapshot_path = saving_path / 'snapshots' / f'snap-{global_step}.pth'
                torch.save(evaluation_model.state_dict(), snapshot_path)
                reasons = []
                if is_best:
                    reasons.append('sampled-val-best')
                if is_periodic:
                    reasons.append('periodic-fullcloud-candidate')
                log_out(f"saved checkpoint={snapshot_path} reason={'+'.join(reasons)}", log_file)
            history.append(miou)
            torch.save({'epoch': epoch, 'global_step': global_step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'ema_state_dict': ema_model.state_dict() if ema_model is not None else None, 'mIou_list': history}, checkpoint_path)
            scheduler.step()
            log_out(f'best sampled-val mIoU={max(history):.3f}', log_file)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return saving_path

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', choices=('train', 'val', 'test', 'vis'), default='train')
    parser.add_argument('--model-path', default=None, help='required for --mode val/test')
    parser.add_argument('--resume', default=None, help='existing ZAHA result directory to resume')
    parser.add_argument('--processed-root', default=None, help='override data/ZAHA/processed/grid_0.200')
    args = parser.parse_args(argv)
    setup_seed(99999)
    dataset = ZAHA(args.processed_root)
    if args.mode == 'train':
        model = CascadeMambaNet(cfg)
        print(f'+ Number of params: {sum((p.numel() for p in model.parameters() if p.requires_grad)) / 1000000.0:.2f}M')
        train_network(model, dataset, args.resume)
        return
    if args.mode == 'vis':
        generator = dataset.get_batch_gen('train')()
        sample = next(generator)
        print('ZAHA sample:', [np.asarray(item).shape for item in sample])
        batch = cascade_collate_fn([sample], device=torch.device('cpu'))
        for stage_idx in range(4):
            print(f"stage {stage_idx}: {tuple(batch[f'features_s{stage_idx}'].shape)}")
        return
    if not args.model_path:
        raise ValueError('--model-path is required for validation or official test; random-weight testing is forbidden')
    from evaluation.zaha_evaluator import ModelTester
    model = CascadeMambaNet(cfg)
    tester = ModelTester(model, dataset, args.model_path)
    tester.test(split='val' if args.mode == 'val' else 'test')
if __name__ == '__main__':
    main()
