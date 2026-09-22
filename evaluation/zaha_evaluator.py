from __future__ import annotations
import gc
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from configs.settings import ZAHAConfig as cfg
from datasets.zaha.metadata import CLASS_NAMES, NUM_CLASSES
from utils.ply import write_ply

class InfiniteDataset(IterableDataset):

    def __init__(self, generator_factory):
        self.generator_factory = generator_factory

    def __iter__(self):
        yield from self.generator_factory()

def log_out(message: str, log_file) -> None:
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

def metrics_from_confusion(confusion: np.ndarray) -> dict:
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = support + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=union > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    f1 = np.divide(2.0 * true_positive, support + predicted, out=np.zeros_like(true_positive), where=support + predicted > 0)
    present = support > 0
    return {'OA': float(true_positive.sum() / max(support.sum(), 1.0)), 'mPrecision': float(precision[present].mean()), 'mRecall': float(recall[present].mean()), 'mF1': float(f1[present].mean()), 'mIoU': float(iou[present].mean()), 'mAcc': float(recall[present].mean()), 'per_class_iou': {name: float(score) for (name, score) in zip(CLASS_NAMES, iou)}, 'per_class_precision': {name: float(score) for (name, score) in zip(CLASS_NAMES, precision)}, 'per_class_recall': {name: float(score) for (name, score) in zip(CLASS_NAMES, recall)}, 'per_class_f1': {name: float(score) for (name, score) in zip(CLASS_NAMES, f1)}, 'per_class_accuracy': {name: float(score) for (name, score) in zip(CLASS_NAMES, recall)}, 'confusion_matrix': confusion.tolist()}

class ModelTester:

    def __init__(self, model, dataset, restore_snapshot: str | Path):
        self.dataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        restore_snapshot = Path(restore_snapshot).expanduser().resolve()
        if not restore_snapshot.exists():
            raise FileNotFoundError(f'model snapshot does not exist: {restore_snapshot}')
        checkpoint = torch.load(restore_snapshot, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            checkpoint = checkpoint.get('ema_state_dict') or checkpoint['model_state_dict']
        model.load_state_dict(checkpoint, strict=True)
        self.model = model.to(self.device).eval()
        self.restore_snapshot = restore_snapshot

    def _create_output(self, split: str) -> tuple[Path, object]:
        stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())
        run_name = f'{cfg.experiment_name}_{split}_{stamp}'
        output_root = Path(cfg.results_root) / 'evaluations' / run_name
        prediction_dir = output_root / 'predictions'
        probability_dir = output_root / '_probabilities'
        output_root.mkdir(parents=True, exist_ok=False)
        prediction_dir.mkdir()
        probability_dir.mkdir()
        Path(cfg.logs_root).mkdir(parents=True, exist_ok=True)
        log_file = (Path(cfg.logs_root) / f'{run_name}.log').open('w', encoding='utf-8')
        return (output_root, log_file)

    def test(self, split: str='test') -> Path:
        if split not in {'val', 'test'}:
            raise ValueError('full-cloud evaluator accepts only val or official test')
        (output_root, log_file) = self._create_output(split)
        prediction_dir = output_root / 'predictions'
        probability_dir = output_root / '_probabilities'
        records = self.dataset.records[split]
        probability_buffers = []
        probability_paths = []
        for record in records:
            path = probability_dir / f"{record['id']}.f32"
            buffer = np.memmap(path, dtype=np.float32, mode='w+', shape=(record['sub_points'], NUM_CLASSES))
            buffer[:] = 0.0
            probability_buffers.append(buffer)
            probability_paths.append(path)
        from engine.zaha_pipeline import cascade_collate_fn
        generator_factory = self.dataset.get_batch_gen(split)
        loader = DataLoader(InfiniteDataset(generator_factory), batch_size=cfg.val_batch_size, collate_fn=cascade_collate_fn)
        log_out(f'split={split}', log_file)
        log_out(f'snapshot={self.restore_snapshot}', log_file)
        log_out('Official one-coverage protocol without test-time augmentation', log_file)
        last_minimum = -0.5
        step = 0
        smoothing = 0.95
        with torch.no_grad():
            for batch_data in loader:
                probabilities = F.softmax(self.model(batch_data), dim=-1).cpu().numpy()
                point_indices = batch_data['input_inds'].cpu().numpy()
                cloud_indices = batch_data['cloud_inds'].cpu().numpy()
                for batch_index in range(probabilities.shape[0]):
                    cloud_index = int(cloud_indices[batch_index, 0])
                    indices = point_indices[batch_index]
                    buffer = probability_buffers[cloud_index]
                    buffer[indices] = smoothing * buffer[indices] + (1.0 - smoothing) * probabilities[batch_index]
                step += 1
                if step % 50 == 0:
                    log_out(f'step={step} min_possibility={np.min(self.dataset.min_possibility[split]):.4f}', log_file)
                current_minimum = float(np.min(self.dataset.min_possibility[split]))
                if current_minimum > 0.5 and current_minimum > last_minimum:
                    last_minimum = current_minimum
                    break
        if last_minimum <= 0.5:
            raise RuntimeError('ZAHA test ended before one complete spatial coverage')
        log_out(f'one complete coverage reached at min_possibility={last_minimum:.4f}', log_file)
        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        prediction_chunks = 0
        for (cloud_index, record) in enumerate(records):
            for (chunk_id, projection, labels) in self.dataset.iter_projection_chunks(split, cloud_index):
                probabilities = np.asarray(probability_buffers[cloud_index][projection])
                prediction = probabilities.argmax(axis=1).astype(np.int32)
                if prediction.size != labels.size:
                    raise RuntimeError(f"projection mismatch for {record['id']} chunk {chunk_id}")
                encoded = NUM_CLASSES * labels + prediction
                confusion += np.bincount(encoded, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
                write_ply(str(prediction_dir / f"{record['id']}__{chunk_id}.ply"), [prediction, labels.astype(np.int32)], ['pred', 'label'])
                prediction_chunks += 1
        metrics = metrics_from_confusion(confusion)
        metrics.update({'split': split, 'protocol': 'LoFG3, one complete coverage, no test-time augmentation', 'snapshot': str(self.restore_snapshot), 'prediction_sources': len(records), 'prediction_chunks': prediction_chunks})
        with (output_root / 'metrics.json').open('w', encoding='utf-8') as handle:
            json.dump(metrics, handle, indent=2)
        log_out(f"OA={metrics['OA'] * 100:.2f} mP={metrics['mPrecision'] * 100:.2f} mR={metrics['mRecall'] * 100:.2f} mF1={metrics['mF1'] * 100:.2f} mIoU={metrics['mIoU'] * 100:.2f}", log_file)
        log_out('IoU: ' + ' '.join((f'{name}={score * 100:.2f}' for (name, score) in metrics['per_class_iou'].items())), log_file)
        log_file.close()
        probabilities = None
        prediction = None
        encoded = None
        buffer = None
        for (index, probability_buffer) in enumerate(probability_buffers):
            probability_buffer.flush()
            mmap_handle = getattr(probability_buffer, '_mmap', None)
            if mmap_handle is not None:
                mmap_handle.close()
            probability_buffers[index] = None
        probability_buffers.clear()
        gc.collect()
        cleanup_error = None
        for _attempt in range(10):
            try:
                shutil.rmtree(probability_dir)
                cleanup_error = None
                break
            except OSError as error:
                cleanup_error = error
                gc.collect()
                time.sleep(0.2)
        if cleanup_error is not None:
            print(f'warning: could not remove scratch directory {probability_dir}: {cleanup_error}')
        return output_root
