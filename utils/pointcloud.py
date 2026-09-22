import numpy as np
from scipy.spatial import cKDTree

class DataProcessing:

    @staticmethod
    def knn_search(support_points, query_points, k):
        batch_size = support_points.shape[0]
        indices = np.zeros((batch_size, query_points.shape[1], k), dtype=np.int32)
        for batch_index in range(batch_size):
            tree = cKDTree(support_points[batch_index])
            (_, neighbors) = tree.query(query_points[batch_index], k=k, workers=-1)
            neighbors = np.asarray(neighbors, dtype=np.int32)
            if k == 1:
                neighbors = neighbors.reshape(-1, 1)
            indices[batch_index] = neighbors
        return indices

    @staticmethod
    def shuffle_idx(values):
        indices = np.arange(len(values))
        np.random.shuffle(indices)
        return values[indices]

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1):
        voxel_indices = np.floor(points / grid_size).astype(np.int64)
        (_, first_indices, inverse) = np.unique(voxel_indices, axis=0, return_index=True, return_inverse=True)
        voxel_count = len(first_indices)
        counts = np.bincount(inverse, minlength=voxel_count).astype(np.float64)
        subsampled_points = np.zeros((voxel_count, points.shape[1]), dtype=np.float64)
        np.add.at(subsampled_points, inverse, points)
        subsampled_points = (subsampled_points / counts[:, None]).astype(np.float32)
        subsampled_features = None
        if features is not None:
            subsampled_features = np.zeros((voxel_count, features.shape[1]), dtype=np.float64)
            np.add.at(subsampled_features, inverse, features)
            subsampled_features = subsampled_features / counts[:, None]
        subsampled_labels = None
        if labels is not None:
            flat_labels = np.asarray(labels).reshape(-1).astype(np.int64)
            label_count = int(flat_labels.max()) + 1
            encoded = inverse.astype(np.int64) * label_count + flat_labels
            votes = np.bincount(encoded, minlength=voxel_count * label_count).reshape(voxel_count, label_count)
            subsampled_labels = votes.argmax(axis=1).astype(np.asarray(labels).dtype)
        if features is None and labels is None:
            return subsampled_points
        if labels is None:
            return (subsampled_points, subsampled_features)
        if features is None:
            return (subsampled_points, subsampled_labels)
        return (subsampled_points, subsampled_features, subsampled_labels)

    @staticmethod
    def get_bf_class_weights():
        counts = np.array([569265, 1843263, 3574466, 10393913, 1490909, 7591659, 7637344, 1270869], dtype=np.float64)
        frequencies = counts / counts.sum()
        return (1.0 / (frequencies + 0.02))[np.newaxis, :].astype(np.float32)

    @staticmethod
    def get_zaha_class_weights(class_counts, smoothing=0.02, max_weight=10.0):
        counts = np.asarray(class_counts, dtype=np.float64).reshape(-1)
        if counts.size != 15:
            raise ValueError(f'expected 15 ZAHA class counts, got {counts.size}')
        if np.any(counts <= 0):
            raise ValueError(f'training data is missing ZAHA classes {np.flatnonzero(counts <= 0).tolist()}')
        frequencies = counts / counts.sum()
        weights = np.minimum(1.0 / (frequencies + float(smoothing)), float(max_weight))
        weights /= weights.mean()
        return weights.astype(np.float32)

    @staticmethod
    def IoU_from_confusions(confusions):
        true_positives = np.diagonal(confusions, axis1=-2, axis2=-1)
        false_positives = np.sum(confusions, axis=-2) - true_positives
        false_negatives = np.sum(confusions, axis=-1) - true_positives
        denominator = true_positives + false_positives + false_negatives + 1e-06
        iou = true_positives / denominator
        mask = np.sum(confusions, axis=-1) < 0.001
        class_count = np.sum(1 - mask, axis=-1, keepdims=True)
        mean_iou = np.sum(iou, axis=-1, keepdims=True) / (class_count + 1e-06)
        return iou + mask * mean_iou
