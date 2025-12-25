import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_classification
from collections import Counter


def UnderALLKNN(request, data, label):

    k_min = int(request.form.get("ALLKNN_k_min"))
    k_max = int(request.form.get("ALLKNN_k_max"))
    if k_max < k_min:
        k_max = k_min
    label = label.ravel()
    data, label = all_knn(data, label, k_min=k_min, k_max=k_max)

    return data, label


def all_knn(X, y, k_min=3, k_max=10, sampling_strategy='auto'):
    """
    实现 AllKNN 欠采样算法，支持字典类型的 sampling_strategy

    参数:
    X: 特征矩阵
    y: 类别标签
    k_min: 最小邻居数，控制第一个 k 值
    k_max: 最大邻居数，控制最后一个 k 值
    sampling_strategy: 欠采样策略，'auto'表示自动处理，字典类型表示指定不同类别的采样数或比例

    返回:
    X_resampled: 欠采样后的特征矩阵
    y_resampled: 欠采样后的类别标签
    """
    # 初始化变量
    X_resampled, y_resampled = X, y
    num_samples = X_resampled.shape[0]

    # 如果指定了 sampling_strategy 为字典类型
    if isinstance(sampling_strategy, dict):
        # 根据字典策略确定每个类别的目标样本数或比例
        target_samples = {}
        for label, strategy in sampling_strategy.items():
            # 如果是整数，表示目标样本数
            if isinstance(strategy, int):
                target_samples[label] = strategy
            # 如果是浮动比例，表示目标样本比例
            elif isinstance(strategy, float):
                target_samples[label] = int(strategy * np.sum(y == label))

    # 遍历从 k_min 到 k_max 的不同 k 值
    for k in range(k_min, k_max + 1):
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_resampled)

        keep_indices = []

        for i in range(X_resampled.shape[0]):
            # 获取当前样本 x_i 的 k 个最近邻
            distances, indices = nn.kneighbors([X_resampled[i]])

            # 获取最近邻的类别标签
            nearest_labels = y_resampled[indices[0]]

            # 当前样本的标签
            current_label = y_resampled[i]

            # 如果当前样本标签在邻居中占多数，则保留该样本
            if np.sum(nearest_labels == current_label) > k // 2:
                keep_indices.append(i)

        # 生成当前 k 值下的欠采样数据集
        new_X_resampled = X_resampled[keep_indices]
        new_y_resampled = y_resampled[keep_indices]

        # 如果需要根据字典策略调整采样，调整每个类别的数量
        if isinstance(sampling_strategy, dict):
            # 获取每个类别的目标样本数，并调整数据集
            resampled_counts = dict(zip(*np.unique(new_y_resampled, return_counts=True)))
            for label, target_count in target_samples.items():
                if label in resampled_counts and resampled_counts[label] > target_count:
                    # 降低该类别的样本数，直到达到目标
                    indices_to_keep = np.where(new_y_resampled == label)[0]
                    indices_to_remove = np.random.choice(indices_to_keep, size=len(indices_to_keep) - target_count,
                                                         replace=False)
                    new_X_resampled = np.delete(new_X_resampled, indices_to_remove, axis=0)
                    new_y_resampled = np.delete(new_y_resampled, indices_to_remove, axis=0)

        # 如果数据集没有变化，说明迭代可以停止
        if np.array_equal(X_resampled, new_X_resampled) and np.array_equal(y_resampled, new_y_resampled):
            break

        # 更新 X_resampled 和 y_resampled
        X_resampled, y_resampled = new_X_resampled, new_y_resampled

    return X_resampled, y_resampled


if __name__ == '__main__':
    # 创建一个示例数据集，目标类别的比例更高
    X, y = make_classification(n_samples=2000, n_features=10, n_classes=5,
                               n_clusters_per_class=1, n_redundant=2, weights=[0.4, 0.3, 0.1, 0.1, 0.1],
                               n_informative=3, random_state=0)

    # 输出原始数据集的类别分布
    print("原始数据集类别分布:", dict(zip(*np.unique(y, return_counts=True))))

    # 应用 AllKNN 进行欠采样
    X_resampled, y_resampled = all_knn(X, y, k_min=4, k_max=5)

    # 输出欠采样后的数据集大小
    print("欠采样后数据集大小:", X_resampled.shape)

    # 输出欠采样后类别的分布
    print("欠采样后数据集类别分布:", dict(zip(*np.unique(y_resampled, return_counts=True))))
