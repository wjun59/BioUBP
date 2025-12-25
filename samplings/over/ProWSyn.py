import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter


def ProWSyn(X, y, sampling_strategy, k_neighbors=5):
    """
    ProWSyn算法的Python实现，支持按照指定策略进行过采样。

    参数:
    X: 输入数据，形状为 (n_samples, n_features)
    y: 标签，形状为 (n_samples,)
    sampling_strategy: 字典形式，指定每个类标签的目标数量
    k_neighbors: 用于原型选择的邻居数目

    返回:
    X_resampled: 过采样后的数据
    y_resampled: 过采样后的标签
    """

    # Step 1: 提取每个类的样本
    unique_classes = np.unique(y)
    class_samples = {cls: X[y == cls] for cls in unique_classes}

    # Step 2: 初始化合成数据集
    X_resampled = X.copy()
    y_resampled = y.copy()

    # Step 3: 根据采样策略生成样本
    for cls in unique_classes:
        target_count = sampling_strategy.get(cls, len(class_samples[cls]))  # 获取目标数量
        current_count = len(class_samples[cls])  # 当前样本数量

        if target_count > current_count:
            # 需要进行过采样
            n_samples_to_generate = target_count - current_count
            synthetic_samples = generate_synthetic_samples(class_samples[cls], n_samples_to_generate, k_neighbors)

            # 添加合成样本到结果中
            X_resampled = np.vstack([X_resampled, synthetic_samples])
            y_resampled = np.hstack([y_resampled, [cls] * n_samples_to_generate])

    return X_resampled, y_resampled


def generate_synthetic_samples(minority_samples, n_samples, k_neighbors):
    """
    根据给定的少数类样本生成合成样本。

    参数:
    minority_samples: 少数类样本
    n_samples: 需要生成的合成样本数量
    k_neighbors: 用于选择原型样本的邻居数量

    返回:
    synthetic_samples: 生成的合成样本
    """

    # 计算少数类样本之间的距离
    knn = NearestNeighbors(n_neighbors=k_neighbors)
    knn.fit(minority_samples)
    distances, indices = knn.kneighbors(minority_samples)

    # 选择原型样本
    n_minority = minority_samples.shape[0]
    synthetic_samples = []

    for _ in range(n_samples):
        prototype_idx = np.random.randint(0, n_minority)
        prototype = minority_samples[prototype_idx]

        # 随机选择一个邻居样本
        neighbor_idx = np.random.choice(indices[prototype_idx][1:])  # 排除自己
        neighbor = minority_samples[neighbor_idx]

        # 在原型和邻居之间进行线性插值生成合成样本
        alpha = np.random.rand()
        synthetic_sample = prototype + alpha * (neighbor - prototype)
        synthetic_samples.append(synthetic_sample)

    return np.array(synthetic_samples)


def OverProWSyn(request, data, label):
    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)
    for l, n in label_counts.items():
        num = int(request.form.get(f'ProWSyn{l}'))
        label_count_dict[l] = num
    print(label_count_dict)
    k_neighbors = int(request.form.get("ProWSyn_k"))
    labels = label.ravel()
    data, label = ProWSyn(data, labels, label_count_dict, k_neighbors=k_neighbors)

    # 每个类的类别数不要少于分类数
    return data, label



if __name__ == '__main__':
    # 生成示例数据（纯分类特征）
    data = np.random.choice([0, 1, 2], size=(15, 3))  # 三列分类特征，每列包含0/1/2三种类别
    labels = np.array([0] * 6 + [1] * 5 + [2] * 4)  # 标签

    # 设定采样策略
    sampling_strategy = {0: 10, 1: 7, 2: 11}

    # 调用 ProWSyn 函数进行过采样
    resampled_data, resampled_labels = ProWSyn(data, labels, sampling_strategy, k_neighbors=3)

    print("Resampled labels:", Counter(resampled_labels))
    print("Resampled data:", resampled_data)
