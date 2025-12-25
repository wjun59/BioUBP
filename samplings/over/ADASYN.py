import numpy as np
from sklearn.neighbors import NearestNeighbors

"""
使用ADASYN（自适应合成采样）进行数据集过采样的函数。

适用场景：
- 该算法适用于二分类和多分类问题，通过生成合成样本来平衡类别分布，尤其关注难以分类的少数类样本。

参数：
- data: ndarray，输入的特征数据，形状为(n_samples, n_features)。
- labels: ndarray，输入的标签，形状为(n_samples,)，包含每个样本对应的类别标签。
- target_class: int，少数类标签，用于生成合成样本。
- n_neighbors: int，选择的近邻数量，用于生成合成样本，默认为5。        小于总的样本数。
- beta: float，控制合成样本生成的比例，范围为 (0, 1](该类和其他类总和的比例)
- random_state: int，随机种子，确保结果可重复，默认为None。

返回值：
- resampled_data: ndarray，包含原始数据和生成的合成样本数据。
- resampled_labels: ndarray，包含原始标签和生成的合成样本标签。
"""


def adasyn_oversample(data, labels, target_class, n_neighbors=5, beta=1.0, random_state=None):
    np.random.seed(random_state)

    minority_data = data[labels == target_class]
    majority_data = data[labels != target_class]

    n_minority = len(minority_data)
    n_majority = len(majority_data)

    # 保证生成至少一个样本
    G = max(1, int((n_majority - n_minority) * beta))

    if G <= 0:
        print("不需要生成合成样本。")
        return data, labels

    print(f"需要生成的合成样本数量: {G}")

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(data)
    neighbors = nn.kneighbors(minority_data, return_distance=False)

    distribution_ratio = np.zeros(n_minority)
    for i in range(n_minority):
        num_majority_neighbors = sum(labels[neighbors[i][1:]] != target_class)
        distribution_ratio[i] = num_majority_neighbors / n_neighbors

    distribution_ratio /= np.sum(distribution_ratio)

    synthetic_samples = []
    for i in range(n_minority):
        num_synthetic_samples = max(1, int(G * (distribution_ratio[i] if not np.isnan(distribution_ratio[i]) else 0)))  # 保证至少生成一个样本
        for _ in range(num_synthetic_samples):
            neighbor_index = np.random.choice(neighbors[i][1:])
            diff = data[neighbor_index] - minority_data[i]
            synthetic_sample = minority_data[i] + np.random.rand() * diff
            synthetic_samples.append(synthetic_sample)

    synthetic_samples = np.array(synthetic_samples)

    if len(synthetic_samples) == 0:
        print("未生成合成样本。")
        return data, labels

    print("生成的合成样本:")
    print(synthetic_samples)

    resampled_data = np.vstack([data, synthetic_samples])
    resampled_labels = np.hstack([labels, np.full(len(synthetic_samples), target_class)])

    return resampled_data, resampled_labels


def Overadasyn(request, data, label):

    target_class = int(request.form.get(f'adasyn_target_class'))
    n_neighbors = int(request.form.get(f'adasyn_n'))
    beta = float(request.form.get(f'adasyn_beta'))

    labels = label.ravel()
    data, label = adasyn_oversample(data, labels, target_class, n_neighbors, beta)

    return data, label


if __name__ == '__main__':
    np.random.seed(42)
    data_class_0 = np.random.normal(loc=1, scale=0.2, size=(10, 3))
    data_class_1 = np.random.normal(loc=5, scale=0.2, size=(5, 3))
    data_class_2 = np.random.normal(loc=3, scale=0.2, size=(1, 3))

    # 将数据和标签合并
    data = np.vstack((data_class_0, data_class_1))
    data = np.vstack((data, data_class_2))
    labels = np.array([0] * 10 + [1] * 5 + [2] * 1)

    # 调用 kpca_oversample 函数进行过采样
    resampled_data, resampled_labels = adasyn_oversample(data, labels, 2, 16, 0.1)

    print(resampled_labels)
    print(resampled_data)
