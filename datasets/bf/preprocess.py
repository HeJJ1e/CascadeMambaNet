from pathlib import Path
import pickle
import numpy as np
from sklearn.neighbors import KDTree
from configs.settings import BFConfig as cfg
from utils.ply import write_ply
from utils.pointcloud import DataProcessing as DP

ROOT = Path(__file__).resolve().parent
RAW_ROOT = cfg.data_root / 'ZHC_Building_Facade'
ORIGINAL_ROOT = cfg.data_root / 'original_ply'
SUBSAMPLED_ROOT = cfg.data_root / f'input_{cfg.sub_grid_size:.3f}'


def compute_normals_and_geometry(points, tree):
    k = min(cfg.normal_k, len(points))
    neighbor_indices = tree.query(points, k=k, return_distance=False)
    neighbors = points[neighbor_indices].astype(np.float32)
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', centered, centered) / k
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0].astype(np.float32)
    normals[normals[:, 2] < 0] *= -1
    normals[(np.abs(normals[:, 2]) < 1e-6) & (normals[:, 0] < 0)] *= -1
    eigenvalues = np.maximum(eigenvalues.astype(np.float32), 0.0)[:, ::-1]
    first, second, third = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    denominator = np.maximum(first, 1e-6)
    height = points[:, 2].astype(np.float32)
    relative_height = (height - height.min()) / max(float(height.max() - height.min()), 1e-6)
    geometry = np.stack([(first - second) / denominator, (second - third) / denominator, third / denominator, 1.0 - np.abs(normals[:, 2]), relative_height], axis=1)
    return normals, np.nan_to_num(geometry, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def convert(annotation_path, class_to_label):
    parts = []
    for path in sorted(annotation_path.glob('*.txt')):
        class_name = path.stem.split('_')[0]
        class_name = class_name if class_name in class_to_label else 'clutter'
        points = np.atleast_2d(np.loadtxt(path, dtype=np.float32))
        labels = np.full((len(points), 1), class_to_label[class_name], dtype=np.float32)
        parts.append(np.concatenate((points, labels), axis=1))
    if not parts:
        raise ValueError(f'no annotation files found in {annotation_path}')
    cloud = np.concatenate(parts, axis=0)
    xyz = cloud[:, :3].astype(np.float32)
    xyz -= xyz.min(axis=0, keepdims=True)
    colors = cloud[:, 3:6].astype(np.uint8)
    labels = cloud[:, 6].astype(np.uint8)
    cloud_name = f'{annotation_path.parent.parent.name}_{annotation_path.parent.name}'
    write_ply(str(ORIGINAL_ROOT / f'{cloud_name}.ply'), (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, cfg.sub_grid_size)
    sub_colors = sub_colors / 255.0
    write_ply(str(SUBSAMPLED_ROOT / f'{cloud_name}.ply'), (sub_xyz, sub_colors, sub_labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
    tree = KDTree(sub_xyz)
    with (SUBSAMPLED_ROOT / f'{cloud_name}_KDTree.pkl').open('wb') as handle:
        pickle.dump(tree, handle, protocol=4)
    projection = tree.query(xyz, k=1, return_distance=False).reshape(-1).astype(np.int32)
    with (SUBSAMPLED_ROOT / f'{cloud_name}_proj.pkl').open('wb') as handle:
        pickle.dump((projection, labels), handle, protocol=4)
    normals, geometry = compute_normals_and_geometry(sub_xyz, tree)
    np.save(SUBSAMPLED_ROOT / f'{cloud_name}_normal.npy', normals)
    np.save(SUBSAMPLED_ROOT / f'{cloud_name}_geo.npy', geometry)
    print(f'prepared {cloud_name}: raw={len(xyz)} subsampled={len(sub_xyz)}')


def main():
    classes = [line.strip() for line in (ROOT / 'classes.txt').read_text().splitlines() if line.strip()]
    class_to_label = {name: index for index, name in enumerate(classes)}
    areas = [line.strip() for line in (ROOT / 'areas.txt').read_text().splitlines() if line.strip()]
    ORIGINAL_ROOT.mkdir(parents=True, exist_ok=True)
    SUBSAMPLED_ROOT.mkdir(parents=True, exist_ok=True)
    for relative_path in areas:
        convert(RAW_ROOT / relative_path, class_to_label)


if __name__ == '__main__':
    main()
