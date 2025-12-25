import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from collections import Counter


def generate_synthetic_samples(X, n_samples, k_neighbors):
    """
    基于 SMOTE 插值生成合成样本，并添加对抗扰动。

    参数:
    X: 输入的少数类样本数据，形状为 (n_samples, n_features)
    n_samples: 需要生成的合成样本数量
    k_neighbors: 用于选择邻居的数量

    返回:
    合成样本，形状为 (n_samples, n_features)
    """
    knn = NearestNeighbors(n_neighbors=k_neighbors)
    knn.fit(X)
    distances, indices = knn.kneighbors(X)

    synthetic_samples = []

    for _ in range(n_samples):
        # 随机选择一个样本
        idx = np.random.randint(0, len(X))
        prototype = X[idx]

        # 随机选择一个邻居
        neighbor_idx = np.random.choice(indices[idx][1:])  # 排除自己
        neighbor = X[neighbor_idx]

        # 进行线性插值
        alpha = np.random.rand()
        synthetic_sample = prototype + alpha * (neighbor - prototype)

        # 对生成的样本添加扰动，增加样本的多样性
        perturbation = np.random.normal(loc=0, scale=0.1, size=synthetic_sample.shape)
        synthetic_sample += perturbation

        # 生成的样本加入列表
        synthetic_samples.append(synthetic_sample)

    return np.array(synthetic_samples)


def RASLE(X, y, sampling_strategy, k_neighbors=5):
    """
    RASLE算法的Python实现，基于SMOTE和对抗性扰动进行过采样。

    参数:
    X: 输入数据，形状为 (n_samples, n_features)
    y: 标签，形状为 (n_samples,)
    sampling_strategy: 字典形式，指定每个类标签的目标数量
    k_neighbors: 用于选择邻居的数量

    返回:
    X_resampled: 过采样后的数据
    y_resampled: 过采样后的标签
    """
    unique_classes = np.unique(y)
    class_samples = {cls: X[y == cls] for cls in unique_classes}

    X_resampled = X.copy()
    y_resampled = y.copy()

    for cls in unique_classes:
        target_count = sampling_strategy.get(cls, len(class_samples[cls]))
        current_count = len(class_samples[cls])

        if target_count > current_count:
            n_samples_to_generate = target_count - current_count
            synthetic_samples = generate_synthetic_samples(class_samples[cls], n_samples_to_generate, k_neighbors)

            # 添加合成样本到结果中
            X_resampled = np.vstack([X_resampled, synthetic_samples])
            y_resampled = np.hstack([y_resampled, [cls] * n_samples_to_generate])

    return X_resampled, y_resampled

def OverRASLE(request, data, label):
    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)
    for l, n in label_counts.items():
        num = int(request.form.get(f'RASLE{l}'))
        label_count_dict[l] = num
    print(label_count_dict)
    k_neighbors = int(request.form.get("RASLE_k"))
    labels = label.ravel()
    data, label = RASLE(data, labels, label_count_dict, k_neighbors=k_neighbors)

    # 每个类的类别数不要少于分类数
    return data, label


# 测试 RASLE 实现
if __name__ == '__main__':
    # 生成示例数据（3类数据）
    data = np.random.choice([0, 1, 2], size=(15, 3))  # 三列分类特征
    labels = np.array([0] * 6 + [1] * 5 + [2] * 4)  # 标签

    # 设定采样策略
    sampling_strategy = {0: 10, 1: 8, 2: 3}

    # 调用 RASLE 函数进行过采样
    resampled_data, resampled_labels = RASLE(data, labels, sampling_strategy, k_neighbors=3)

    print("Resampled labels:", Counter(resampled_labels))
    print("Resampled data:", resampled_data)
