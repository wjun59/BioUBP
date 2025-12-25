from imblearn.over_sampling import SMOTEN
from collections import Counter
import numpy as np

"""
使用SMOTEN进行过采样的函数。

适用场景：
- 该算法适用于仅包含分类特征的数据集。
- 通过生成新样本来平衡类别分布，保持分类特征的一致性。

参数：
- data: ndarray，输入的特征数据，形状为(n_samples, n_features)。
- labels: ndarray，输入的标签，形状为(n_samples,)，包含每个样本对应的类别标签。
- sampling_strategy: dict，采样策略，默认为‘auto’。

- random_state: int，随机种子，确保结果可重复，默认为None。
- k_neighbors: int，查找最近邻的数量，用于合成样本生成，默认为5。

返回值：
- resampled_data: ndarray，生成的合成样本数据。
- resampled_labels: ndarray，生成的合成样本对应的标签。
"""


def smoten_oversample(data, labels, sampling_strategy, random_state=None, k_neighbors=5):
    smote = SMOTEN(
        sampling_strategy=sampling_strategy,  # 采样策略，控制每类的目标样本数量
        random_state=random_state,  # 随机种子，保证结果可重复
        k_neighbors=k_neighbors  # 用于生成合成样本的最近邻个数
    )

    resampled_data, resampled_labels = smote.fit_resample(data, labels)

    return resampled_data, resampled_labels


def OverSMOTEN(request, data, label):
    # 使用 Counter 获取每个唯一值的计数
    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    for l, n in label_counts.items():
        num = int(request.form.get(f'SMOTEN{l}'))
        label_count_dict[l] = num
    print(label_count_dict)

    labels = label.ravel()
    k_neighbors = int(request.form.get("SMOTEN_k"))
    resampled_data, resampled_labels = smoten_oversample(
        data, labels, label_count_dict, k_neighbors=k_neighbors
    )

    return resampled_data, resampled_labels


if __name__ == '__main__':
    # 生成示例数据（纯分类特征）
    data = np.random.choice([0, 1, 2], size=(15, 3))  # 三列分类特征，每列包含0/1/2三种类别
    labels = np.array([0] * 6 + [1] * 5 + [2] * 4)  # 标签

    # 设定采样策略
    # sampling_strategy = {"auto"}
    sampling_strategy = {0: 10, 1: 7, 2: 11}
    # 调用 SMOTEN 函数进行过采样
    resampled_data, resampled_labels = smoten_oversample(
        data, labels, sampling_strategy, k_neighbors=3
    )

    print("Resampled labels:", Counter(resampled_labels))
    print("Resampled data:", resampled_data)
