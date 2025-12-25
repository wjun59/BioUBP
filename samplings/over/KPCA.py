import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import NearestNeighbors
from collections import Counter


"""
使用KPCA（核主成分分析）进行过采样的函数。

适用场景：
- 该算法适用于二分类和多分类问题，通过生成合成样本来平衡类别分布。

参数：
- data: ndarray，输入的特征数据，形状为(n_samples, n_features)。
- labels: ndarray，输入的标签，形状为(n_samples,)，包含每个样本对应的类别标签。
- sampling_strategy: dict，采样策略，以标签为键，对应的目标样本数量为值。
- n_components: int，KPCA中的主成分数，默认为2。
- kernel: str，使用的核函数类型，默认为'rbf'。
- gamma: float，核函数的参数，默认为None。如果为None，将使用默认值。
- random_state: int，随机种子，确保结果可重复，默认为None。

返回值：
- resampled_data: ndarray，生成的合成样本数据。
- resampled_labels: ndarray，生成的合成样本对应的标签。
"""


def kpca_oversample(data, labels, sampling_strategy, n_components=2, kernel='rbf', gamma=None, random_state=None):
    import numpy as np
    from sklearn.decomposition import KernelPCA
    from sklearn.neighbors import NearestNeighbors

    resampled_data = []
    resampled_labels = []

    labels = np.array(labels, dtype=int)

    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, fit_inverse_transform=True,
                     random_state=random_state)

    for label, target_count in sampling_strategy.items():
        indices = np.where(labels == label)[0]
        data_label = data[indices]

        # 在当前标签的数据上拟合KPCA
        kpca.fit(data_label)

        # 使用KPCA组件生成合成样本
        synthetic_samples = kpca.inverse_transform(kpca.transform(data_label))

        # 需要的合成样本数量
        num_synthetic_samples = target_count - len(indices)

        if num_synthetic_samples > 0:
            nn = NearestNeighbors(n_neighbors=min(len(indices), 5), algorithm='auto', metric='euclidean')
            nn.fit(data_label)

            neighbors = nn.kneighbors(data_label, return_distance=False)

            # 控制生成的合成样本数
            interpolation_weights = np.random.rand(num_synthetic_samples, len(neighbors[0]))
            interpolation_weights /= interpolation_weights.sum(axis=1)[:, np.newaxis]

            additional_synthetic_samples = []

            for i in range(num_synthetic_samples):
                neighbor_indices = neighbors[np.random.choice(len(data_label))]
                neighbor_points = data_label[neighbor_indices]
                synthetic_sample = np.dot(interpolation_weights[i], neighbor_points)
                additional_synthetic_samples.append(synthetic_sample)

            additional_synthetic_samples = np.array(additional_synthetic_samples)
            synthetic_samples = np.vstack((synthetic_samples, additional_synthetic_samples))

        # 添加抽样后的数据和标签，确保只使用所需数量的合成样本
        resampled_data.append(synthetic_samples[:target_count])
        resampled_labels.extend([label] * target_count)

    resampled_data = np.concatenate(resampled_data, axis=0)
    resampled_labels = np.array(resampled_labels)

    return resampled_data, resampled_labels


def OverKPCA(request, data, label):
    n_components = int(request.form.get(f'kpca_n'))  # 2-3
    # 使用 Counter 获取每个唯一值的计数
    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    for l, n in label_counts.items():
        num = int(request.form.get(f'KPCA{l}'))
        label_count_dict[l] = num
    print(label_count_dict)

    label = label.ravel()
    data, label = kpca_oversample(data, label, label_count_dict, n_components)

    return data, label



if __name__ == '__main__' :
    np.random.seed(42)
    data_class_0 = np.random.normal(loc=1, scale=0.2, size=(10, 2))
    data_class_1 = np.random.normal(loc=5, scale=0.2, size=(5, 2))
    data_class_2 = np.random.normal(loc=3, scale=0.2, size=(5, 2))

    # 将数据和标签合并
    data = np.vstack((data_class_0, data_class_1))
    data = np.vstack((data, data_class_2))
    labels = np.array([0] * 10 + [1] * 5 + [2]*1)

    # 设定采样策略：将类别 1 扩增到 10 个样本
    sampling_strategy = {0: 10, 1: 10, 2: 10}

    # 调用 kpca_oversample 函数进行过采样
    resampled_data, resampled_labels = kpca_oversample(
        data, labels, sampling_strategy, n_components=3, kernel='rbf', gamma=0.1, random_state=42
    )

    print(resampled_labels)
    print(resampled_data)
