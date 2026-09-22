from __future__ import annotations
import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Iterator
import numpy as np
from sklearn.neighbors import KDTree
from datasets.zaha.metadata import CLASS_NAMES, NUM_CLASSES, RAW_PCD_ROOT, SPLITS, processed_root, to_train_labels
from utils.ply import read_ply, write_ply
RAW_TILE_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('label', 'u1')])

def _parse_pcd_header(handle, path: Path) -> tuple[list[str], int | None]:
    fields: list[str] | None = None
    points: int | None = None
    while True:
        line = handle.readline()
        if not line:
            raise ValueError(f'{path}: unexpected EOF in PCD header')
        text = line.decode('ascii', errors='strict').strip()
        if not text or text.startswith('#'):
            continue
        parts = text.split()
        keyword = parts[0].upper()
        if keyword == 'FIELDS':
            fields = parts[1:]
        elif keyword == 'POINTS':
            if len(parts) != 2:
                raise ValueError(f'{path}: malformed POINTS header')
            points = int(parts[1])
        elif keyword == 'DATA':
            if len(parts) != 2 or parts[1].lower() != 'ascii':
                raise ValueError(f'{path}: only ASCII PCD is supported')
            break
    if fields is None:
        raise ValueError(f'{path}: missing FIELDS header')
    required = {'x', 'y', 'z', 'classification', 'rgb'}
    missing = required - set(fields)
    if missing:
        raise ValueError(f'{path}: missing required PCD fields {sorted(missing)}')
    return (fields, points)

def iter_pcd_chunks(path: Path, chunk_points: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if chunk_points <= 0:
        raise ValueError('chunk_points must be positive')
    with path.open('rb') as handle:
        (fields, expected_points) = _parse_pcd_header(handle, path)
        field_count = len(fields)
        (x_idx, y_idx, z_idx) = (fields.index(axis) for axis in ('x', 'y', 'z'))
        label_idx = fields.index('classification')
        read_points = 0
        while True:
            lines = []
            for _ in range(chunk_points):
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    lines.append(line)
            if not lines:
                break
            values = np.fromstring(b''.join(lines).decode('ascii'), sep=' ', dtype=np.float64)
            if values.size % field_count:
                raise ValueError(f'{path}: point rows do not match {field_count} declared fields')
            rows = values.reshape(-1, field_count)
            raw_labels = rows[:, label_idx]
            rounded_labels = np.rint(raw_labels).astype(np.int32)
            if not np.allclose(raw_labels, rounded_labels):
                raise ValueError(f'{path}: non-integral classification values encountered')
            labels = to_train_labels(rounded_labels)
            xyz = rows[:, [x_idx, y_idx, z_idx]].astype(np.float32, copy=False)
            if not np.isfinite(xyz).all():
                raise ValueError(f'{path}: non-finite coordinates encountered')
            read_points += len(xyz)
            yield (xyz, labels)
        if expected_points is not None and read_points != expected_points:
            raise ValueError(f'{path}: header says {expected_points} points but parsed {read_points}')

def _tile_name(tx: int, ty: int, tz: int) -> str:
    return f'tile_x{tx:+07d}_y{ty:+07d}_z{tz:+07d}'

def partition_source_cloud(pcd_path: Path, temp_dir: Path, tile_size: float, chunk_points: int) -> tuple[dict[str, Path], np.ndarray, int]:
    if tile_size <= 0:
        raise ValueError('tile_size must be positive')
    temp_dir.mkdir(parents=True, exist_ok=True)
    raw_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_points = 0
    tile_paths: dict[str, Path] = {}
    for (xyz, labels) in iter_pcd_chunks(pcd_path, chunk_points):
        tile_xyz = np.floor(xyz / tile_size).astype(np.int64)
        (unique_tiles, inverse) = np.unique(tile_xyz, axis=0, return_inverse=True)
        raw_counts += np.bincount(labels, minlength=NUM_CLASSES)
        total_points += len(labels)
        for (local_idx, (tx, ty, tz)) in enumerate(unique_tiles):
            mask = inverse == local_idx
            name = _tile_name(int(tx), int(ty), int(tz))
            tile_path = temp_dir / f'{name}.bin'
            records = np.empty(int(mask.sum()), dtype=RAW_TILE_DTYPE)
            records['x'] = xyz[mask, 0]
            records['y'] = xyz[mask, 1]
            records['z'] = xyz[mask, 2]
            records['label'] = labels[mask]
            with tile_path.open('ab') as handle:
                records.tofile(handle)
            tile_paths[name] = tile_path
    return (tile_paths, raw_counts, total_points)

def voxel_subsample(xyz: np.ndarray, labels: np.ndarray, grid_size: float) -> tuple[np.ndarray, np.ndarray]:
    if len(xyz) == 0:
        raise ValueError('cannot voxel-subsample an empty tile')
    voxel = np.floor(xyz / grid_size).astype(np.int64)
    (_, inverse) = np.unique(voxel, axis=0, return_inverse=True)
    n_voxels = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=n_voxels).astype(np.float64)
    sub_xyz = np.column_stack([np.bincount(inverse, weights=xyz[:, axis], minlength=n_voxels) / counts for axis in range(3)]).astype(np.float32)
    encoded = inverse.astype(np.int64) * NUM_CLASSES + labels.astype(np.int64)
    votes = np.bincount(encoded, minlength=n_voxels * NUM_CLASSES).reshape(n_voxels, NUM_CLASSES)
    sub_labels = votes.argmax(axis=1).astype(np.uint8)
    return (sub_xyz, sub_labels)

def compute_normals_and_geometry(points: np.ndarray, tree: KDTree, normal_k: int, batch_points: int) -> tuple[np.ndarray, np.ndarray]:
    n_points = len(points)
    if n_points == 0:
        raise ValueError('cannot compute features for an empty tile')
    k = min(max(int(normal_k), 1), n_points)
    normals = np.empty((n_points, 3), dtype=np.float32)
    geometry = np.empty((n_points, 5), dtype=np.float32)
    (z_min, z_max) = (float(points[:, 2].min()), float(points[:, 2].max()))
    z_span = max(z_max - z_min, 1e-06)
    for start in range(0, n_points, batch_points):
        end = min(start + batch_points, n_points)
        query_points = points[start:end]
        neigh_idx = tree.query(query_points, k=k, return_distance=False)
        neighs = points[neigh_idx].astype(np.float32, copy=False)
        centered = neighs - neighs.mean(axis=1, keepdims=True)
        cov = np.einsum('nki,nkj->nij', centered, centered, optimize=True) / float(k)
        (eigvals, eigvecs) = np.linalg.eigh(cov)
        normal = eigvecs[:, :, 0].astype(np.float32, copy=False)
        normal[normal[:, 2] < 0] *= -1.0
        tie = (np.abs(normal[:, 2]) < 1e-06) & (normal[:, 0] < 0)
        normal[tie] *= -1.0
        eig = np.maximum(eigvals[:, ::-1], 0.0).astype(np.float32, copy=False)
        denom = np.maximum(eig[:, 0], 1e-06)
        features = np.stack([(eig[:, 0] - eig[:, 1]) / denom, (eig[:, 1] - eig[:, 2]) / denom, eig[:, 2] / denom, 1.0 - np.abs(normal[:, 2]), (query_points[:, 2] - z_min) / z_span], axis=1)
        normals[start:end] = np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
        geometry[start:end] = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return (normals, geometry)

def projection_indices(tree: KDTree, raw_xyz: np.ndarray, batch_points: int) -> np.ndarray:
    proj = np.empty(len(raw_xyz), dtype=np.int32)
    for start in range(0, len(raw_xyz), batch_points):
        end = min(start + batch_points, len(raw_xyz))
        proj[start:end] = tree.query(raw_xyz[start:end], k=1, return_distance=False).reshape(-1).astype(np.int32, copy=False)
    return proj

def read_raw_tile(raw_tile_path: Path, max_tile_points: int) -> tuple[np.ndarray, np.ndarray]:
    raw_bytes = raw_tile_path.stat().st_size
    if raw_bytes % RAW_TILE_DTYPE.itemsize:
        raise ValueError(f'{raw_tile_path}: corrupted temporary tile record size')
    raw_points = raw_bytes // RAW_TILE_DTYPE.itemsize
    if raw_points == 0:
        raise ValueError(f'{raw_tile_path}: empty temporary tile')
    if raw_points > max_tile_points:
        raise MemoryError(f'{raw_tile_path} contains {raw_points:,} raw points; reduce --tile-voxels or raise --max-tile-points after confirming server RAM.')
    records = np.fromfile(raw_tile_path, dtype=RAW_TILE_DTYPE)
    xyz = np.column_stack((records['x'], records['y'], records['z'])).astype(np.float32)
    labels = records['label'].astype(np.uint8, copy=False)
    return (xyz, labels)

def voxelize_raw_tile(raw_tile_path: Path, grid_size: float, max_tile_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (xyz, labels) = read_raw_tile(raw_tile_path, max_tile_points)
    (sub_xyz, sub_labels) = voxel_subsample(xyz, labels, grid_size)
    if len(sub_xyz) < 1:
        raise ValueError(f'{raw_tile_path}: no voxelised points')
    return (xyz, labels, sub_xyz, sub_labels)

def process_source_cloud(raw_tile_paths: dict[str, Path], output_dir: Path, source_id: str, split: str, grid_size: float, normal_k: int, feature_batch_points: int, max_tile_points: int) -> dict:
    ordered_tiles = sorted(raw_tile_paths.items())
    if not ordered_tiles:
        raise ValueError(f'{source_id}: no temporary raw tiles')
    sub_xyz_parts: list[np.ndarray] = []
    sub_label_parts: list[np.ndarray] = []
    raw_sizes: dict[str, int] = {}
    for (tile_name, raw_tile_path) in ordered_tiles:
        (xyz, labels, sub_xyz, sub_labels) = voxelize_raw_tile(raw_tile_path=raw_tile_path, grid_size=grid_size, max_tile_points=max_tile_points)
        raw_sizes[tile_name] = int(len(xyz))
        sub_xyz_parts.append(sub_xyz)
        sub_label_parts.append(sub_labels)
        if split == 'train':
            raw_tile_path.unlink()
    sub_xyz = np.concatenate(sub_xyz_parts, axis=0).astype(np.float32, copy=False)
    sub_labels = np.concatenate(sub_label_parts, axis=0).astype(np.uint8, copy=False)
    if len(sub_xyz) < 1:
        raise ValueError(f'{source_id}: no voxelised points after merging tiles')
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    ply_path = split_dir / f'{source_id}.ply'
    tree_path = split_dir / f'{source_id}_KDTree.pkl'
    normal_path = split_dir / f'{source_id}_normal.npy'
    geo_path = split_dir / f'{source_id}_geo.npy'
    write_ply(str(ply_path), [sub_xyz, sub_labels], ['x', 'y', 'z', 'class'])
    tree = KDTree(sub_xyz, leaf_size=50)
    with tree_path.open('wb') as handle:
        pickle.dump(tree, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (normals, geometry) = compute_normals_and_geometry(sub_xyz, tree, normal_k, feature_batch_points)
    np.save(normal_path, normals)
    np.save(geo_path, geometry)
    record = {'id': source_id, 'source_id': source_id, 'split': split, 'ply': str(ply_path.relative_to(output_dir)), 'tree': str(tree_path.relative_to(output_dir)), 'normal': str(normal_path.relative_to(output_dir)), 'geo': str(geo_path.relative_to(output_dir)), 'raw_points': int(sum(raw_sizes.values())), 'sub_points': int(len(sub_xyz))}
    if split in {'val', 'test'}:
        projection_dir = split_dir / 'projections' / source_id
        projection_dir.mkdir(parents=True, exist_ok=True)
        projection_paths: list[str] = []
        for (tile_name, raw_tile_path) in ordered_tiles:
            (xyz, labels) = read_raw_tile(raw_tile_path=raw_tile_path, max_tile_points=max_tile_points)
            proj_path = projection_dir / f'{tile_name}_proj.pkl'
            proj = projection_indices(tree, xyz, feature_batch_points)
            with proj_path.open('wb') as handle:
                pickle.dump([proj, labels], handle, protocol=pickle.HIGHEST_PROTOCOL)
            projection_paths.append(str(proj_path.relative_to(output_dir)))
            raw_tile_path.unlink()
        record['projection_chunks'] = projection_paths
    return record

def prepare(args: argparse.Namespace) -> Path:
    if args.tile_voxels <= 0:
        raise ValueError('tile_voxels must be positive')
    if args.grid_size <= 0:
        raise ValueError('grid_size must be positive')
    tile_size = args.grid_size * args.tile_voxels
    output_dir = processed_root(args.grid_size)
    manifest_path = output_dir / 'manifest.json'
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f'{output_dir} already exists. Use --overwrite only to rebuild this ZAHA grid.')
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_root = output_dir / '_temporary_raw_tiles'
    manifest = {'dataset': 'ZAHA', 'label_protocol': 'LoFG3', 'num_classes': NUM_CLASSES, 'class_names': list(CLASS_NAMES), 'label_mapping': 'PCD classification 1..15 -> network label 0..14', 'rgb_policy': 'discarded_label_palette_never_model_input', 'sampling_unit': 'source_level_global_voxel_cloud', 'temporary_partition': 'XYZ tiles used only to bound raw preprocessing memory', 'grid_size': args.grid_size, 'tile_voxels': args.tile_voxels, 'tile_size': tile_size, 'normal_k': args.normal_k, 'splits': {split: [] for split in SPLITS}, 'raw_class_counts': {split: [0] * NUM_CLASSES for split in SPLITS}, 'subsampled_class_counts': {split: [0] * NUM_CLASSES for split in SPLITS}}
    try:
        for split in args.splits:
            source_dir = RAW_PCD_ROOT / split
            pcd_paths = sorted(source_dir.glob('*.pcd'))
            if not pcd_paths:
                raise FileNotFoundError(f'no PCD files found in {source_dir}')
            if args.limit is not None:
                pcd_paths = pcd_paths[:args.limit]
            for pcd_path in pcd_paths:
                source_id = pcd_path.stem
                source_temp = temp_root / split / source_id
                print(f'[{split}] partitioning {pcd_path.name}')
                (tile_paths, raw_counts, total_points) = partition_source_cloud(pcd_path, source_temp, tile_size, args.chunk_points)
                manifest['raw_class_counts'][split] = (np.asarray(manifest['raw_class_counts'][split], dtype=np.int64) + raw_counts).tolist()
                print(f'[{split}] {pcd_path.name}: {total_points:,} raw points, {len(tile_paths)} spatial tiles')
                record = process_source_cloud(raw_tile_paths=tile_paths, output_dir=output_dir, source_id=source_id, split=split, grid_size=args.grid_size, normal_k=args.normal_k, feature_batch_points=args.feature_batch_points, max_tile_points=args.max_tile_points)
                manifest['splits'][split].append(record)
                counts = np.bincount(np.asarray(read_ply(str(output_dir / record['ply']))['class'], dtype=np.int64), minlength=NUM_CLASSES)
                manifest['subsampled_class_counts'][split] = (np.asarray(manifest['subsampled_class_counts'][split], dtype=np.int64) + counts).tolist()
                shutil.rmtree(source_temp)
    finally:
        if temp_root.exists() and (not any(temp_root.rglob('*.bin'))):
            shutil.rmtree(temp_root)
    manifest['split_source_counts'] = {split: len(manifest['splits'][split]) for split in SPLITS}
    manifest['split_subsampled_points'] = {split: int(sum((item['sub_points'] for item in manifest['splits'][split]))) for split in SPLITS}
    with manifest_path.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
    print(f'Wrote {manifest_path}')
    return manifest_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid-size', type=float, default=0.2)
    parser.add_argument('--tile-voxels', type=int, default=128, help='XYZ tile width in voxels; 128 with a 0.20 m grid gives 25.6 m cubic tiles.')
    parser.add_argument('--normal-k', type=int, default=20)
    parser.add_argument('--chunk-points', type=int, default=1000000)
    parser.add_argument('--feature-batch-points', type=int, default=200000)
    parser.add_argument('--max-tile-points', type=int, default=10000000)
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=list(SPLITS))
    parser.add_argument('--limit', type=int, default=None, help='process only the first N sources per split')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()
if __name__ == '__main__':
    prepare(parse_args())
