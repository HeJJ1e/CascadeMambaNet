import argparse
import copy
import glob
import math
import os
import pickle
import random
import time
from os import makedirs
from os.path import exists, join

def _configure_cuda_visibility_from_cli():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpu', type=int, default=0)
    arguments, _ = parser.parse_known_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(arguments.gpu)

_configure_cuda_visibility_from_cli()

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.amp import autocast
from torch.utils.data import DataLoader, IterableDataset
from configs.settings import BFConfig as cfg
from evaluation.bf_evaluator import ModelTester
from models.cascade_mamba_net import CascadeMambaNet
from models.losses import lovasz_softmax
from utils.ply import read_ply
from utils.pointcloud import DataProcessing as DP

def cbl_loss(logits, labels, neigh_idx, ignore_idx, temperature, num_classes):
    (B, N) = labels.shape
    K = neigh_idx.shape[-1]
    logits_flat = logits.reshape(B * N, num_classes).float()
    labels_flat = labels.reshape(B * N).long()
    batch_off = (torch.arange(B, device=labels.device) * N).view(B, 1, 1)
    neigh_idx_glb = (neigh_idx + batch_off).reshape(B * N, K).long()
    feat = F.normalize(logits_flat, dim=-1)
    neigh_feat = feat[neigh_idx_glb]
    neigh_label = labels_flat[neigh_idx_glb]
    valid_self = labels_flat != ignore_idx
    valid_neigh = neigh_label != ignore_idx
    posmask = (neigh_label == labels_flat.unsqueeze(-1)) & valid_neigh
    has_pos = posmask.any(dim=-1)
    has_neg = (~posmask & valid_neigh).any(dim=-1)
    boundary = has_pos & has_neg & valid_self
    if not boundary.any():
        return logits_flat.sum() * 0.0
    sim = (feat.unsqueeze(1) * neigh_feat).sum(-1)
    sim_b = sim[boundary]
    posmask_b = posmask[boundary].float()
    valid_neigh_b = valid_neigh[boundary].float()
    sim_b = sim_b / temperature
    sim_b = sim_b - sim_b.max(dim=-1, keepdim=True)[0]
    exp_sim = torch.exp(sim_b) * valid_neigh_b
    pos = (exp_sim * posmask_b).sum(dim=-1)
    neg = exp_sim.sum(dim=-1)
    return (-torch.log((pos + 1e-12) / (neg + 1e-12))).mean()

def _compute_normals_and_geometry(points, search_tree, k=20):
    k = min(int(k), points.shape[0])
    (_, neigh_idx) = search_tree.query(points, k=k)
    if k == 1:
        neigh_idx = neigh_idx[:, None]
    neighs = points[neigh_idx].astype(np.float32)
    mean = neighs.mean(axis=1, keepdims=True)
    centered = neighs - mean
    cov = np.einsum('nki,nkj->nij', centered, centered) / k
    (eigvals, eigvecs) = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0].astype(np.float32)
    flip = normals[:, 2] < 0
    normals[flip] = -normals[flip]
    tie = (np.abs(normals[:, 2]) < 1e-06) & (normals[:, 0] < 0)
    normals[tie] = -normals[tie]
    eigvals = np.maximum(eigvals.astype(np.float32), 0.0)
    eig_desc = eigvals[:, ::-1]
    (l1, l2, l3) = (eig_desc[:, 0], eig_desc[:, 1], eig_desc[:, 2])
    denom = np.maximum(l1, 1e-06)
    linearity = (l1 - l2) / denom
    planarity = (l2 - l3) / denom
    scattering = l3 / denom
    verticality = 1.0 - np.abs(normals[:, 2])
    z = points[:, 2].astype(np.float32)
    relative_height = (z - z.min()) / max(float(z.max() - z.min()), 1e-06)
    geometry = np.stack([linearity, planarity, scattering, verticality, relative_height], axis=1)
    geometry = np.nan_to_num(geometry, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return (normals, geometry)

def _compute_normals(points, search_tree, k=20):
    (normals, _) = _compute_normals_and_geometry(points, search_tree, k)
    return normals

class ModelEMA:

    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for (ema_p, p) in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.detach().data, alpha=1.0 - self.decay)
        for (ema_b, b) in zip(self.ema.buffers(), model.buffers()):
            ema_b.data.copy_(b.data)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)

def cascade_collate_fn(batch):
    batch_xyz = np.stack([item[0] for item in batch])
    batch_colors = np.stack([item[1] for item in batch])
    batch_labels = np.stack([item[2] for item in batch])
    batch_pc_idx = np.stack([item[3] for item in batch])
    batch_cloud_idx = np.stack([item[4] for item in batch])
    (B, N, _) = batch_xyz.shape
    if len(batch[0]) >= 6:
        batch_normals = np.stack([item[5] for item in batch])
    else:
        batch_normals = np.zeros((B, N, 3), dtype=np.float32)
    if len(batch[0]) >= 7:
        batch_geos = np.stack([item[6] for item in batch])
    else:
        batch_geos = np.zeros((B, N, getattr(cfg, 'geo_feature_dim', 5)), dtype=np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = {}
    strides = [16, 8, 2, 1]
    stage_xyz_list = []
    for s_idx in range(4):
        stride = strides[s_idx]
        if stride == 1:
            s_xyz = batch_xyz
            s_colors = batch_colors
            s_labels = batch_labels
            s_normals = batch_normals if batch_normals is not None else None
            s_geos = batch_geos if batch_geos is not None else None
        else:
            sel = np.arange(0, N, stride)
            s_xyz = batch_xyz[:, sel, :]
            s_colors = batch_colors[:, sel, :]
            s_labels = batch_labels[:, sel]
            s_normals = batch_normals[:, sel, :] if batch_normals is not None else None
            s_geos = batch_geos[:, sel, :] if batch_geos is not None else None
        stage_xyz_list.append(s_xyz)
        feat_parts = [s_xyz, s_colors]
        feat_parts.append(s_normals)
        feat_parts.append(s_geos)
        s_features = np.concatenate(feat_parts, axis=-1)
        s_neigh_idx = DP.knn_search(s_xyz, s_xyz, cfg.k_n)
        inputs[f'xyz_s{s_idx}'] = torch.from_numpy(s_xyz).float().to(device)
        inputs[f'neigh_idx_s{s_idx}'] = torch.from_numpy(s_neigh_idx).long().to(device)
        inputs[f'features_s{s_idx}'] = torch.from_numpy(s_features).float().to(device)
        inputs[f'labels_s{s_idx}'] = torch.from_numpy(s_labels).long().to(device)
    for s_idx in range(1, 4):
        prev_xyz = stage_xyz_list[s_idx - 1]
        curr_xyz = stage_xyz_list[s_idx]
        upsample_idx = DP.knn_search(prev_xyz, curr_xyz, k=3)
        inputs[f'upsample_idx_s{s_idx}'] = torch.from_numpy(upsample_idx).long().to(device)
    inputs['labels'] = torch.from_numpy(batch_labels).long().to(device)
    inputs['input_inds'] = torch.from_numpy(batch_pc_idx).long().to(device)
    inputs['cloud_inds'] = torch.from_numpy(batch_cloud_idx).long().to(device)
    return inputs

class PointCloudIterableDataset(IterableDataset):

    def __init__(self, generator_func, num_per_epoch):
        self.generator_func = generator_func
        self.num_per_epoch = num_per_epoch

    def __iter__(self):
        gen = self.generator_func()
        for (i, item) in enumerate(gen):
            if i >= self.num_per_epoch:
                break
            yield item

class BF:

    def __init__(self, test_area_idx):
        self.name = 'BF'
        self.path = str(cfg.data_root)
        self.label_to_names = {0: 'balustrade', 1: 'balcony', 2: 'advboard', 3: 'wall', 4: 'eave', 5: 'column', 6: 'window', 7: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for (k, v) in self.label_to_names.items()])
        self.label_to_idx = {l: i for (i, l) in enumerate(self.label_values)}
        self.ignored_labels = np.array([])
        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply'))
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.input_normals = {'training': [], 'validation': []}
        self.input_geos = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for (i, file_path) in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))
            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]
            normal_file = join(tree_path, '{:s}_normal.npy'.format(cloud_name))
            geo_file = join(tree_path, '{:s}_geo.npy'.format(cloud_name))
            sub_normals = np.load(normal_file) if exists(normal_file) else None
            sub_geos = np.load(geo_file) if exists(geo_file) else None
            if sub_normals is None or sub_geos is None:
                pts_raw = np.array(search_tree.data, copy=False).astype(np.float32)
                (computed_normals, computed_geos) = _compute_normals_and_geometry(pts_raw, search_tree, cfg.normal_k)
            if sub_normals is None:
                sub_normals = computed_normals
                np.save(normal_file, sub_normals)
            if sub_geos is None:
                sub_geos = computed_geos
                np.save(geo_file, sub_geos)
            self.input_normals[cloud_split] += [sub_normals.astype(np.float32, copy=False)]
            self.input_geos[cloud_split] += [sub_geos.astype(np.float32, copy=False)]
            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-06, time.time() - t0))
        print('\nPreparing reprojected indices for testing')
        for (i, file_path) in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    (proj_idx, labels) = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        self.possibility[split] = []
        self.min_possibility[split] = []
        for (i, tree) in enumerate(self.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 0.001]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]
        def _sample_one():
            cloud_idx = int(np.argmin(self.min_possibility[split]))
            point_ind = np.argmin(self.possibility[split][cloud_idx])
            points = np.array(self.input_trees[split][cloud_idx].data, copy=False)
            center_point = points[point_ind, :].reshape(1, -1)
            noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
            pick_point = center_point + noise.astype(center_point.dtype)
            if len(points) < cfg.num_points:
                queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
            else:
                queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]
            queried_idx = DP.shuffle_idx(queried_idx)
            queried_pc_xyz = points[queried_idx]
            queried_pc_xyz = queried_pc_xyz - pick_point
            queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
            queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]
            queried_pc_normals = self.input_normals[split][cloud_idx][queried_idx]
            queried_pc_geos = self.input_geos[split][cloud_idx][queried_idx]
            dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
            delta = np.square(1 - dists / np.max(dists))
            self.possibility[split][cloud_idx][queried_idx] += delta
            self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))
            if len(points) < cfg.num_points:
                num_in = len(queried_pc_xyz)
                dup = np.random.choice(num_in, cfg.num_points - num_in)
                pad_idx = np.concatenate([np.arange(num_in), dup])
                queried_pc_xyz = queried_pc_xyz[pad_idx]
                queried_pc_colors = queried_pc_colors[pad_idx]
                queried_idx = queried_idx[pad_idx]
                queried_pc_labels = queried_pc_labels[pad_idx]
                queried_pc_normals = queried_pc_normals[pad_idx]
                queried_pc_geos = queried_pc_geos[pad_idx]
            if split == 'training':
                theta = np.random.uniform(0, 2 * np.pi)
                (cos_t, sin_t) = (np.cos(theta), np.sin(theta))
                rot = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
                queried_pc_xyz = queried_pc_xyz @ rot.T
                queried_pc_normals = queried_pc_normals @ rot.T
                scale = np.random.uniform(0.9, 1.1)
                queried_pc_xyz = queried_pc_xyz * scale
                queried_pc_xyz += np.random.normal(0, 0.01, queried_pc_xyz.shape).astype(np.float32)
                if np.random.random() < 0.1:
                    queried_pc_colors = np.zeros_like(queried_pc_colors)
            return (queried_pc_xyz.astype(np.float32), queried_pc_colors.astype(np.float32), queried_pc_labels, queried_idx.astype(np.int32), np.array([cloud_idx], dtype=np.int32), queried_pc_normals.astype(np.float32), queried_pc_geos.astype(np.float32))

        def spatially_regular_gen():
            use_mix3d = split == 'training'
            mix3d_prob = cfg.mix3d_prob
            mix3d_gap = cfg.mix3d_gap
            while True:
                if use_mix3d and np.random.random() < mix3d_prob:
                    (xyz1, c1, l1, qi1, ci1, n1, g1) = _sample_one()
                    (xyz2, c2, l2, _, _, n2, g2) = _sample_one()
                    x_shift = float(xyz1[:, 0].max() - xyz2[:, 0].min()) + mix3d_gap
                    xyz2 = xyz2.copy()
                    xyz2[:, 0] += x_shift
                    xyz_all = np.concatenate([xyz1, xyz2], axis=0)
                    c_all = np.concatenate([c1, c2], axis=0)
                    l_all = np.concatenate([l1, l2], axis=0)
                    n_all = np.concatenate([n1, n2], axis=0)
                    g_all = np.concatenate([g1, g2], axis=0)
                    sel = np.random.choice(len(xyz_all), cfg.num_points, replace=False)
                    yield (xyz_all[sel].astype(np.float32), c_all[sel].astype(np.float32), l_all[sel], qi1[:cfg.num_points], ci1, n_all[sel].astype(np.float32), g_all[sel].astype(np.float32))
                else:
                    yield _sample_one()
        return spatially_regular_gen

def evaluate(model, val_loader, cfg, log_file):
    model.eval()
    gt_classes = [0 for _ in range(cfg.num_classes)]
    positive_classes = [0 for _ in range(cfg.num_classes)]
    true_positive_classes = [0 for _ in range(cfg.num_classes)]
    val_total_correct = 0
    val_total_seen = 0
    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        for (step_id, batch_data) in enumerate(val_loader):
            if step_id % 50 == 0:
                print(f'{step_id} / {cfg.val_steps}')
            logits = model(batch_data)
            labels = batch_data['labels']
            preds = torch.argmax(logits, dim=-1)
            pred_valid = preds.cpu().numpy().flatten()
            labels_valid = labels.cpu().numpy().flatten()
            if hasattr(cfg, 'ignored_label_inds') and len(cfg.ignored_label_inds) > 0:
                valid_idx = np.in1d(labels_valid, cfg.ignored_label_inds, invert=True)
                pred_valid = pred_valid[valid_idx]
                labels_valid = labels_valid[valid_idx]
            correct = np.sum(pred_valid == labels_valid)
            val_total_correct += correct
            val_total_seen += len(labels_valid)
            conf_matrix = confusion_matrix(labels_valid, pred_valid, labels=np.arange(0, cfg.num_classes, 1))
            gt_classes += np.sum(conf_matrix, axis=1)
            positive_classes += np.sum(conf_matrix, axis=0)
            true_positive_classes += np.diagonal(conf_matrix)
    iou_list = []
    for n in range(0, cfg.num_classes, 1):
        denominator = float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
        iou = true_positive_classes[n] / denominator if denominator > 0 else 0.0
        iou_list.append(iou)
    mean_iou = sum(iou_list) / float(cfg.num_classes) * 100
    acc = val_total_correct / float(val_total_seen)
    log_out(f'eval accuracy: {acc:.4f}', log_file)
    log_out(f'Mean IoU = {mean_iou:.1f}%', log_file)
    s = f'{mean_iou:5.2f} | '
    for IoU in iou_list:
        s += f'{100 * IoU:5.2f} '
    log_out('-' * len(s), log_file)
    log_out(s, log_file)
    log_out('-' * len(s) + '\n', log_file)
    return mean_iou

def train_network(model, dataset, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if cfg.saving_path is None:
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
    else:
        saving_path = cfg.saving_path
    makedirs(saving_path) if not exists(saving_path) else None
    log_file = open(join(saving_path, f'log_train_{dataset.name}_{dataset.val_split}.txt'), 'a')
    train_gen = dataset.get_batch_gen('training')
    val_gen = dataset.get_batch_gen('validation')
    train_ds = PointCloudIterableDataset(train_gen, cfg.train_steps * cfg.batch_size)
    val_ds = PointCloudIterableDataset(val_gen, cfg.val_steps * cfg.val_batch_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, collate_fn=cascade_collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.val_batch_size, collate_fn=cascade_collate_fn, drop_last=True)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.0005)

    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.max_epoch - cfg.warmup_epochs)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = cfg.eta_min / cfg.learning_rate
        return min_factor + (1.0 - min_factor) * cos_factor
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    ema_model = ModelEMA(model, decay=cfg.ema_decay)
    class_weights = DP.get_bf_class_weights()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).squeeze(0).to(device)
    ignore_index = cfg.ignored_label_inds[0] if hasattr(cfg, 'ignored_label_inds') and len(cfg.ignored_label_inds) > 0 else -100
    focal_gamma = 2.0
    lovasz_w = cfg.lovasz_weight
    stage_weights = cfg.stage_weights
    mIou_list = [0.0]
    global_step = 1
    start_epoch = 0
    ckpt_path = join(saving_path, 'last_ckpt.pth')
    if hasattr(cfg, '_resume') and cfg._resume and exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if ema_model is not None and 'ema_state_dict' in ckpt:
            ema_model.load_state_dict(ckpt['ema_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        mIou_list = ckpt['mIou_list']
        log_out(f'=== Resumed from epoch {start_epoch}, global_step {global_step}, best mIoU {max(mIou_list):.3f} ===', log_file)
    for epoch in range(start_epoch, cfg.max_epoch):
        log_out(f'****EPOCH {epoch}****', log_file)
        model.train()
        for (step, batch_data) in enumerate(train_loader):
            t_start = time.time()
            optimizer.zero_grad()
            with autocast('cuda', dtype=torch.bfloat16):
                model_out = model(batch_data)
            preds_list = model_out
            total_loss = 0.0
            for (s_idx, pred) in enumerate(preds_list):
                s_labels = batch_data[f'labels_s{s_idx}']
                logits_flat = pred.float().reshape(-1, cfg.num_classes)
                labels_flat = s_labels.reshape(-1)
                neigh_idx_s = batch_data[f'neigh_idx_s{s_idx}']
                (B_s, N_s) = s_labels.shape
                neigh_labels = s_labels.gather(1, neigh_idx_s.reshape(B_s, -1)).reshape(B_s, N_s, -1)
                is_boundary = (neigh_labels != s_labels.unsqueeze(-1)).any(dim=-1).float()
                boundary_flat = is_boundary.reshape(-1)
                if ignore_index >= 0:
                    valid_mask = labels_flat != ignore_index
                    logits_flat = logits_flat[valid_mask]
                    labels_flat = labels_flat[valid_mask]
                    if boundary_flat is not None:
                        boundary_flat = boundary_flat[valid_mask]
                ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction='none')
                p_t = torch.exp(-ce_loss)
                focal_weight = (1.0 - p_t) ** focal_gamma
                one_hot = F.one_hot(labels_flat, num_classes=cfg.num_classes).float()
                per_sample_weight = (class_weights_tensor * one_hot).sum(dim=1)
                if boundary_flat is not None:
                    boundary_w = 1.0 + (cfg.boundary_weight - 1.0) * boundary_flat
                    focal_loss = (focal_weight * per_sample_weight * ce_loss * boundary_w).mean()
                else:
                    focal_loss = (focal_weight * per_sample_weight * ce_loss).mean()
                lov_loss = lovasz_softmax(logits_flat, labels_flat, classes='present')
                stage_loss = (1.0 - lovasz_w) * focal_loss + lovasz_w * lov_loss
                total_loss += stage_weights[s_idx] * stage_loss
                if s_idx in getattr(cfg, 'cbl_stages', ()):
                    cbl_l = cbl_loss(pred, s_labels, batch_data[f'neigh_idx_s{s_idx}'], ignore_index, cfg.cbl_temperature, cfg.num_classes)
                    total_loss = total_loss + cfg.cbl_weight * cbl_l
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            if ema_model is not None:
                ema_model.update(model)
            t_end = time.time()
            if global_step % 50 == 0:
                with torch.no_grad():
                    last_pred = torch.argmax(preds_list[-1], dim=-1)
                    last_labels = batch_data['labels']
                    acc = (last_pred == last_labels).float().mean().item()
                message = f'Step {global_step:08d} L_out={total_loss.item():5.3f} Acc={acc:4.2f} ---{1000 * (t_end - t_start):8.2f} ms/batch'
                log_out(message, log_file)
            global_step += 1
        eval_target = ema_model.ema if ema_model is not None else model
        m_iou = evaluate(eval_target, val_loader, cfg, log_file)
        torch.cuda.empty_cache()
        if m_iou > max(mIou_list):
            snapshot_directory = join(saving_path, 'snapshots')
            makedirs(snapshot_directory) if not exists(snapshot_directory) else None
            torch.save(eval_target.state_dict(), join(snapshot_directory, f'snap-{global_step}.pth'))
        mIou_list.append(m_iou)
        log_out(f'Best m_IoU is: {max(mIou_list):5.3f}', log_file)
        ckpt_dict = {'epoch': epoch, 'global_step': global_step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'mIou_list': mIou_list}
        if ema_model is not None:
            ckpt_dict['ema_state_dict'] = ema_model.state_dict()
        torch.save(ckpt_dict, join(saving_path, 'last_ckpt.pth'))
        scheduler.step()
    log_file.close()

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--mode', choices=('train', 'test'), default='train')
    parser.add_argument('--model-path', '--model_path', dest='model_path', default=None)
    parser.add_argument('--resume', type=str, default=None, help='resume training from saving_path (e.g. results/Log_xxx)')
    FLAGS = parser.parse_args(argv)
    Mode = FLAGS.mode
    setup_seed(99999)
    test_area = FLAGS.test_area
    dataset = BF(test_area)
    cfg.ignored_label_inds = [dataset.label_to_idx[ign_label] for ign_label in dataset.ignored_labels]
    if Mode == 'train':
        model = CascadeMambaNet(cfg)
        total_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
        print('+ Number of params: %.2fM' % (total_params / 1000000.0))
        if FLAGS.resume:
            cfg.saving_path = FLAGS.resume
            cfg._resume = True
        else:
            cfg._resume = False
        train_network(model, dataset, cfg)
    elif Mode == 'test':
        cfg.saving = False
        model = CascadeMambaNet(cfg)
        restore_snap = None
        if FLAGS.model_path:
            restore_snap = FLAGS.model_path
        else:
            snap_dir = 'results'
            if exists(snap_dir):
                import glob as g
                snap_files = g.glob(join(snap_dir, '**', 'snapshots', 'snap-*.pth'), recursive=True)
                if snap_files:
                    snap_files.sort(key=os.path.getmtime)
                    restore_snap = snap_files[-1]
                    print(f'Auto-detected snapshot: {restore_snap}')
        if restore_snap is None:
            print('Error: No model snapshot found. Use --model_path to specify one.')
        else:
            tester = ModelTester(model, dataset, restore_snap=restore_snap)
            tester.test(model, dataset)
if __name__ == '__main__':
    main()
