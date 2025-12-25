import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_classification
from collections import Counter


def UnderRENN(request, data, label):

    k = int(request.form.get("RENN_k"))
    y = label.ravel()
    data, label = repeated_edited_nearest_neighbors(data, y, k=k)

    return data, label


def repeated_edited_nearest_neighbors(X, y, k=3):
    """
    实现RENN欠采样算法

    参数:
    X: 特征矩阵
    y: 类别标签
    k: 最近邻数 The number of nearest neighbors

    返回:
    X_resampled: 欠采样后的特征矩阵
    y_resampled: 欠采样后的类别标签
    """
    # 初始化变量
    X_resampled, y_resampled = X, y
    changed = True  # 标记是否需要继续迭代

    while changed:
        # 使用ENN对当前数据集进行一次欠采样
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_resampled)

        keep_indices = []

        for i in range(X_resampled.shape[0]):
            # 找到当前样本的k个最近邻
            distances, indices = nn.kneighbors([X_resampled[i]])

            # 获取最近邻的标签
            nearest_labels = y_resampled[indices[0]]

            # 当前样本的标签
            current_label = y_resampled[i]

            # 如果当前样本标签在邻居中占多数，保留样本
            if np.sum(nearest_labels == current_label) > k // 2:
                keep_indices.append(i)

        # 生成新的数据集
        new_X_resampled = X_resampled[keep_indices]
        new_y_resampled = y_resampled[keep_indices]

        # 检查数据集是否发生变化
        if np.array_equal(X_resampled, new_X_resampled) and np.array_equal(y_resampled, new_y_resampled):
            changed = False  # 数据集没有变化，停止迭代

        # 更新数据集
        X_resampled, y_resampled = new_X_resampled, new_y_resampled

    return X_resampled, y_resampled


if __name__ == '__main__':
    # 创建一个示例数据集，目标类别的比例更高
    X, y = make_classification(n_samples=2000, n_features=10, n_classes=5,
                               n_clusters_per_class=1, n_redundant=2, weights=[0.4, 0.3, 0.1, 0.1, 0.1], n_informative=3, random_state=0)

    # 输出原始数据集的类别分布
    print("原始数据集类别分布:", dict(zip(*np.unique(y, return_counts=True))))

    # 应用RENN进行欠采样
    X_resampled, y_resampled = repeated_edited_nearest_neighbors(X, y, k=10)

    # 输出欠采样后的数据集大小
    print("欠采样后数据集大小:", X_resampled.shape)

    # 输出欠采样后类别的分布
    print("欠采样后数据集类别分布:", dict(zip(*np.unique(y_resampled, return_counts=True))))
