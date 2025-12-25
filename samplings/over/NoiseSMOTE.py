import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter


def noise_smote(data, labels, sampling_strategy, noise_factor=0.05, k_neighbors=5, random_state=None):
    """
    NoiseSMOTE 过采样方法。

    参数：
    - data: ndarray，输入的特征数据，形状为 (n_samples, n_features)
    - labels: ndarray，输入的标签，形状为 (n_samples,)
    - sampling_strategy: dict，采样策略，控制每个类的目标样本数目
    - noise_factor: float，噪声因子，控制噪声的幅度，默认 0.05
    - k_neighbors: int，最近邻数目，默认 5
    - random_state: int，随机种子，确保结果可重复，默认为 None

    返回：
    - resampled_data: ndarray，生成的合成样本数据
    - resampled_labels: ndarray，生成的合成样本对应的标签
    """
    # 使用 SMOTE 生成合成样本
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    smote_data, smote_labels = smote.fit_resample(data, labels)

    # 为合成样本添加噪声
    noise = np.random.normal(0, noise_factor, smote_data.shape)  # 生成噪声
    noisy_data = smote_data + noise  # 将噪声加到合成样本上

    return noisy_data, smote_labels


def OverNoiseSMOTE(request, data, label):

    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    for l, n in label_counts.items():
        num = int(request.form.get(f'NoiseSMOTE{l}'))
        label_count_dict[l] = num
    print(label_count_dict)

    k_neighbors = int(request.form.get("NoiseSMOTE_k"))
    NoiseSMOTE_factor = float(request.form.get("NoiseSMOTE_factor"))

    labels = label.ravel()
    resampled_data, resampled_labels = noise_smote(
        data, labels, label_count_dict, noise_factor=NoiseSMOTE_factor, k_neighbors=k_neighbors
    )

    return resampled_data, resampled_labels


# 测试
if __name__ == "__main__":
    # 创建示例数据
    data_class_0 = np.random.normal(loc=1, scale=0.2, size=(10, 10))
    data_class_1 = np.random.normal(loc=2, scale=0.2, size=(5, 10))
    data_class_2 = np.random.normal(loc=3, scale=0.2, size=(3, 10))

    # 合并数据和标签
    data = np.vstack((data_class_0, data_class_1, data_class_2))
    labels = np.array([0] * 10 + [1] * 5 + [2] * 3)

    # 设置采样策略
    sampling_strategy = {0: 15, 1: 10, 2: 100}  # 设定每个类别的目标样本数

    # 调用 NoiseSMOTE 进行过采样
    resampled_data, resampled_labels = noise_smote(
        data, labels, sampling_strategy, noise_factor=0.05, k_neighbors=2
    )

    print("Resampled labels:", Counter(resampled_labels))

