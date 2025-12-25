from imblearn.over_sampling import KMeansSMOTE
from collections import Counter
import numpy as np

"""
使用KMeansSMOTE进行过采样的函数。

适用场景：
- 该算法适用于二分类和多分类问题，结合K-Means聚类与SMOTE生成合成样本来平衡类别分布。

参数：
- data: ndarray，输入的特征数据，形状为(n_samples, n_features)。
- labels: ndarray，输入的标签，形状为(n_samples,)，包含每个样本对应的类别标签。
- sampling_strategy: dict，采样策略，默认为‘auto’。

- random_state: int，随机种子，确保结果可重复，默认为None。
- k_neighbors: int，查找最近邻的数量，用于合成样本生成，默认为3。
- cluster_balance_threshold: float，聚类平衡阈值，默认为0.1。
- n_clusters: int，KMeans的聚类数目，默认为3。

返回值：
- resampled_data: ndarray，生成的合成样本数据。
- resampled_labels: ndarray，生成的合成样本对应的标签。
"""


def kmeans_smote_oversample(data, labels, sampling_strategy, random_state=None,
                            k_neighbors=5, cluster_balance_threshold=0.1, n_clusters=2):

    smote = KMeansSMOTE(
        sampling_strategy=sampling_strategy,  # 采样策略，控制每类的目标样本数量
        random_state=random_state,  # 随机种子，保证结果可重复
        k_neighbors=k_neighbors,  # 用于生成合成样本的最近邻个数
        cluster_balance_threshold=cluster_balance_threshold,  # 聚类平衡阈值
        kmeans_estimator=n_clusters  # KMeans的聚类数量
    )

    resampled_data, resampled_labels = smote.fit_resample(data, labels)

    return resampled_data, resampled_labels


def OverKMeansSMOTE(request, data, label):

    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    for l, n in label_counts.items():
        num = int(request.form.get(f'KMeansSMOTE{l}'))
        label_count_dict[l] = num
    print(label_count_dict)

    k_neighbors = int(request.form.get("KMeansSMOTE_k"))
    n_clusters = int(request.form.get("KMeansSMOTE_n"))
    cluster_balance_threshold = float(request.form.get("KMeansSMOTE_cluster_balance_threshold"))

    labels = label.ravel()
    data, label = kmeans_smote_oversample(
        data, labels, label_count_dict, k_neighbors=k_neighbors, cluster_balance_threshold=cluster_balance_threshold, n_clusters=n_clusters
    )

    # 每个类的类别数不要少于分类数
    return data, label


if __name__ == '__main__':

    data_class_0 = np.random.normal(loc=1, scale=0.2, size=(10, 10))
    data_class_1 = np.random.normal(loc=2, scale=0.2, size=(5, 10))
    data_class_2 = np.random.normal(loc=3, scale=0.2, size=(4, 10))

    # 将数据和标签合并
    data = np.vstack((data_class_0, data_class_1, data_class_2))
    labels = np.array([0] * 10 + [1] * 5 + [2] * 4)

    # 设定采样策略
    sampling_strategy = {0: 10, 1: 7, 2: 11}

    # 调用 KMeansSMOTE 函数进行过采样
    resampled_data, resampled_labels = kmeans_smote_oversample(
        data, labels, sampling_strategy, k_neighbors=3, cluster_balance_threshold=0.5, n_clusters=3
    )

    print("Resampled labels:", Counter(resampled_labels))
