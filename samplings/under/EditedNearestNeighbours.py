import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_classification


def edited_nearest_neighbors(X, y, k=3):
    """
    实现ENN欠采样算法

    参数:
    X: 特征矩阵
    y: 类别标签
    k: 最近邻数

    返回:
    X_resampled: 欠采样后的特征矩阵
    y_resampled: 欠采样后的类别标签
    """
    # 初始化一个最近邻模型
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)

    # 用于存储要保留的样本索引
    keep_indices = []

    for i in range(X.shape[0]):
        # 找到当前样本的k个最近邻
        distances, indices = nn.kneighbors([X[i]])

        # 获取最近邻的标签
        nearest_labels = y[indices[0]]

        # 当前样本的标签
        current_label = y[i]

        # 检查邻居的标签与当前样本的标签是否一致
        # 如果当前样本标签在邻居中少数，删除该样本
        if np.sum(nearest_labels == current_label) > k // 2:
            keep_indices.append(i)

    # 返回欠采样后的数据
    X_resampled = X[keep_indices]
    y_resampled = y[keep_indices]

    return X_resampled, y_resampled


def UnderENN(request, data, label):
    k = int(request.form.get(f'ENN_k'))  # 小于样本数
    label = label.ravel()
    X_resampled, y_resampled = edited_nearest_neighbors(data, label, k=k)
    return X_resampled, y_resampled


if __name__ == '__main__':
    # 创建一个示例数据集，目标类别的比例更高
    X, y = make_classification(n_samples=2000, n_features=10, n_classes=5,
                               n_clusters_per_class=1, n_redundant=2, weights=[0.4, 0.3, 0.1, 0.1, 0.1], n_informative=3, random_state=0)

    # 输出原始数据集的类别分布
    print("原始数据集类别分布:", dict(zip(*np.unique(y, return_counts=True))))

    # 选择类别1进行欠采样
    X_resampled, y_resampled = edited_nearest_neighbors(X, y, k=10)

    # 输出欠采样后的数据集大小
    print("欠采样后数据集大小:", X_resampled.shape)

    # 输出欠采样后类别的分布
    print("欠采样后数据集类别分布:", dict(zip(*np.unique(y_resampled, return_counts=True))))

