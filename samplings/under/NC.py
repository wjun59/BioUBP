from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
import numpy as np


def UnderNC(req, X, y):
    n_neighbors = int(req.form.get(f'NC_n_neighbors'))
    sampling_strategy = req.form.get(f'NC_sampling_strategy')
    resampled_data, resampled_labels = neighborhood_cleaning_resample(X, y, sampling_strategy, n_neighbors)
    return resampled_data, resampled_labels


def neighborhood_cleaning_resample(data, labels, sampling_strategy, n_neighbors):
    """
    使用 Neighborhood Cleaning (NC) 算法进行欠采样。
    参数:
    - data: 特征矩阵，类型为 numpy.ndarray
    - labels: 标签向量，类型为 numpy.ndarray
    - sampling_strategy: 一个字典，定义了每个类的采样策略，例如 {0: 3, 1: 3}
    返回:
    - resampled_data: 经过欠采样的特征矩阵
    - resampled_labels: 经过欠采样的标签向量
    """
    # Step 1: 使用 TomekLinks 删除某些多数类别的样本
    tomek = TomekLinks(sampling_strategy=sampling_strategy)
    data, labels = tomek.fit_resample(data, labels)

    # Step 2: 使用 EditedNearestNeighbours 删除一些边界样本
    enn = EditedNearestNeighbours(sampling_strategy=sampling_strategy, n_neighbors=n_neighbors)
    resampled_data, resampled_labels = enn.fit_resample(data, labels)

    return resampled_data, resampled_labels


if __name__ == '__main__':
    # 生成每个类别的数据
    X = np.random.rand(140, 10)  # 类别 0 的样本
    y = np.array([0] * 50 + [1] * 20 + [2] * 70)
    sampling_strategy = 'all'
    # sampling_strategy = {0: 10, 1: 15, 2: 30}
    # majority', 'all', 'auto', 'not minority', 'not majority'

    # 使用 Neighborhood Cleaning (NC) 算法进行欠采样
    x_resampled, y_resampled = neighborhood_cleaning_resample(X, y, sampling_strategy, n_neighbors=2)
    print(x_resampled.shape)

    unique, counts = np.unique(y_resampled, return_counts=True)
    sampled_distribution = dict(zip(unique, counts))

    print("采样后的类别分布:")
    for label, count in sampled_distribution.items():
        print(f"类别 {label}: {count} 个样本")
