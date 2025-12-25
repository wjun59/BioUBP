import numpy as np
from scipy.spatial.distance import pdist, squareform
from collections import Counter


def KSU(data, labels, sampling_strategy):
    """
    Perform undersampling on the input data based on the given sampling strategy.

    Parameters:
    - data: numpy array, shape (n_samples, n_features), the input data.
    - labels: numpy array, shape (n_samples,), the labels corresponding to the input data.
    - sampling_strategy: dict, specifying the number of samples to retain for each class.

    Returns:
    - resampled_data: numpy array, the resampled data after undersampling.
    - resampled_labels: numpy array, the labels corresponding to the resampled data.
    """

    def undersample_class(class_data, num_samples):
        """
        Perform undersampling on a specific class of data.

        Parameters:
        - class_data: numpy array, shape (n_samples, n_features), the data for a specific class.
        - num_samples: int, the number of samples to retain after undersampling.

        Returns:
        - sampled_data: numpy array, the undersampled data.
        """
        if len(class_data) <= num_samples:
            return class_data

        # Standardize the samples
        standardized_data = class_data / np.max(class_data, axis=0)

        # Compute pairwise Euclidean distances
        distances = squareform(pdist(standardized_data, 'euclidean'))

        # Set the diagonal to infinity to avoid self-selection
        np.fill_diagonal(distances, np.inf)

        # Get the indices of samples to keep
        keep_indices = np.argsort(np.min(distances, axis=1))[:num_samples]

        # Select the samples to keep
        sampled_data = class_data[keep_indices, :]

        return sampled_data

    resampled_data = []
    resampled_labels = []

    for label, num_samples in sampling_strategy.items():
        class_data = data[labels == label].astype(float)
        sampled_class_data = undersample_class(class_data, num_samples)
        resampled_data.append(sampled_class_data)
        resampled_labels.extend([label] * sampled_class_data.shape[0])

    resampled_data = np.vstack(resampled_data)
    resampled_labels = np.array(resampled_labels)

    return resampled_data, resampled_labels


def UnderKSU(req, X, y):
    label_counts = Counter(y.flatten())
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    for l, n in label_counts.items():
        num = int(req.form.get(f'KSU{l}'))
        label_count_dict[l] = num
    print(label_count_dict)
    y = y.ravel()
    data, label = KSU(X, y, label_count_dict)

    return data, label


if __name__ == '__main__':
    # 生成每个类别的数据
    X = np.random.rand(100, 4)  # 类别 0 的样本
    y = np.array([0] * 10 + [1] * 20 + [2] * 70)
    sampling_strategy = {0: 8, 1: 15, 2: 40}

    x_resampled, y_resampled = KSU(X, y, sampling_strategy)
    print(x_resampled.shape)

    unique, counts = np.unique(y_resampled, return_counts=True)
    sampled_distribution = dict(zip(unique, counts))

    print("采样后的类别分布:")
    for label, count in sampled_distribution.items():
        print(f"类别 {label}: {count} 个样本")

