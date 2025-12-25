from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter
import numpy as np

"""
使用Borderline-SMOTE进行过采样的函数。

适用场景：
- 该算法适用于二分类和多分类问题，通过生成合成样本来平衡类别分布。

参数：
- data: ndarray，输入的特征数据，形状为(n_samples, n_features)。
- labels: ndarray，输入的标签，形状为(n_samples,)，包含每个样本对应的类别标签。
- sampling_strategy: dict，采样策略，‘auto‘。

- k_neighbors: int，查找最近邻的数量，用于合成样本生成，默认为5。
- m_neighbors: int，查找边界样本时使用的最近邻样本数量，默认为10。                                                             
- kind: str，BorderlineSMOTE 的类型，可选值为 "borderline-1" 或 "borderline-2"，默认为 "borderline-1"。

返回值：
- resampled_data: ndarray，生成的合成样本数据。
- resampled_labels: ndarray，生成的合成样本对应的标签。
"""


def borderline_smote_oversample(data, labels, sampling_strategy, random_state=None,
                                k_neighbors=5, m_neighbors=10, kind="borderline-1"):

    smote = BorderlineSMOTE(
        sampling_strategy=sampling_strategy,  # 采样策略，控制每类的目标样本数量
        random_state=random_state,  # 随机种子，保证结果可重复
        k_neighbors=k_neighbors,  # 用于生成合成样本的最近邻个数
        m_neighbors=m_neighbors,  # 边界样本检测时的最近邻个数
        kind=kind  # Borderline-SMOTE 类型
    )    # 使用Borderline-SMOTE生成合成样本

    resampled_data, resampled_labels = smote.fit_resample(data, labels)

    return resampled_data, resampled_labels


def OverBorderlineSMOTE(request, data, label):
    # 使用 Counter 获取每个唯一值的计数
    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    sampling_strategy = request.form.get("BSMOTE_sampling_strategy", "auto")

    label = label.ravel()

    # 从表单中获取可调参数
    k_neighbors = int(request.form.get("BSMOTE_k", 3))
    m_neighbors = int(request.form.get("BSMOTE_m", 5))
    kind = request.form.get("BSMOTE_kind", "borderline-1")
    # "borderline-2"：更激进，可能在边界外产生样本
    # "borderline-1"：生成的合成样本在边界附近

    # 调用函数进行过采样
    data, label = borderline_smote_oversample(data, label, sampling_strategy,
                                              k_neighbors=k_neighbors, m_neighbors=m_neighbors, kind=kind)

    # 使用 Counter 获取每个唯一值的计数
    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Now counts:", label_count_dict)

    return data, label


if __name__ == '__main__':

    data_class_0 = np.random.normal(loc=1, scale=0.2, size=(10, 10))
    data_class_1 = np.random.normal(loc=2, scale=0.2, size=(5, 10))
    data_class_2 = np.random.normal(loc=3, scale=0.2, size=(2, 10))

    # 将数据和标签合并
    data = np.vstack((data_class_0, data_class_1, data_class_2))
    labels = np.array([0] * 10 + [1] * 5 + [2] * 2)

    # 设定采样策略
    sampling_strategy = "minority"


    # 调用 BorderlineSMOTE 函数进行过采样
    resampled_data, resampled_labels = borderline_smote_oversample(
        data, labels, sampling_strategy, k_neighbors=3, m_neighbors=6, kind="borderline-2"
    )

    print("Resampled labels:", Counter(resampled_labels))

