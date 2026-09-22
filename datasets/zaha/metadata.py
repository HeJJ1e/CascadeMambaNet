from __future__ import annotations
from pathlib import Path
NUM_CLASSES = 15
CLASS_NAMES = ('wall', 'window', 'door', 'balcony', 'molding', 'deco', 'column', 'arch', 'stairs', 'ground_surface', 'terrain', 'roof', 'blinds', 'interior', 'other')
LOFG2_CLASS_NAMES = ('floor', 'decoration', 'structural', 'opening', 'other_elements')
LOFG2_GROUPS = ((9, 10), (4, 5), (0, 3, 6, 7, 8), (1, 2, 12), (11, 13, 14))
LOFG3_TO_LOFG2 = (2, 3, 3, 2, 1, 1, 2, 2, 2, 0, 0, 4, 3, 4, 4)
if len(LOFG3_TO_LOFG2) != NUM_CLASSES:
    raise RuntimeError('LOFG3_TO_LOFG2 must contain one entry per LoFG3 class')
if tuple(sorted((class_id for group in LOFG2_GROUPS for class_id in group))) != tuple(range(NUM_CLASSES)):
    raise RuntimeError('LOFG2_GROUPS must cover every LoFG3 class exactly once')
RAW_LABEL_TO_TRAIN = {raw: raw - 1 for raw in range(1, NUM_CLASSES + 1)}
SPLITS = ('train', 'val', 'test')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZAHA_DATA_ROOT = PROJECT_ROOT / 'data' / 'ZAHA'
RAW_PCD_ROOT = ZAHA_DATA_ROOT / 'raw_pcd'

def processed_root(grid_size: float) -> Path:
    return ZAHA_DATA_ROOT / 'processed' / f'grid_{grid_size:.3f}'

def to_train_labels(raw_labels):
    import numpy as np
    raw_labels = np.asarray(raw_labels)
    if raw_labels.size and (int(raw_labels.min()) < 1 or int(raw_labels.max()) > NUM_CLASSES):
        bad = raw_labels[(raw_labels < 1) | (raw_labels > NUM_CLASSES)][:10]
        raise ValueError(f'ZAHA classification must be in [1, {NUM_CLASSES}], got {bad.tolist()}')
    return (raw_labels.astype(np.int16, copy=False) - 1).astype(np.uint8, copy=False)
