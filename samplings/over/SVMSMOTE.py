from imblearn.over_sampling import SVMSMOTE
from collections import Counter
import numpy as np

"""
使用SVMSMOTE进行过采样的函数。

适用场景：
- 该算法适用于二分类和多分类问题，通过支持向量机(SVM)找到边界样本并生成合成样本来平衡类别分布。

参数：
- data: ndarray，输入的特征数据，形状为(n_samples, n_features)。
- labels: ndarray，输入的标签，形状为(n_samples,)，包含每个样本对应的类别标签。
- sampling_strategy: dict，采样策略，默认为‘auto’。

- random_state: int，随机种子，确保结果可重复，默认为None。
- k_neighbors: int，查找最近邻的数量，用于合成样本生成，默认为3。
- svm_estimator: object，自定义SVM分类器，默认为None。
- out_step: float，调整生成样本位置的步长，默认为0.5。

返回值：
- resampled_data: ndarray，生成的合成样本数据。
- resampled_labels: ndarray，生成的合成样本对应的标签。
"""


def svm_smote_oversample(data, labels, sampling_strategy, random_state=None,
                         k_neighbors=4, svm_estimator=None, out_step=0.5):

    smote = SVMSMOTE(
        sampling_strategy=sampling_strategy,  # 采样策略，控制每类的目标样本数量
        random_state=random_state,  # 随机种子，保证结果可重复
        k_neighbors=k_neighbors,  # 用于生成合成样本的最近邻个数
        svm_estimator=svm_estimator,  # 自定义SVM分类器
        out_step=out_step  # 调整生成样本位置的步长
    )

    resampled_data, resampled_labels = smote.fit_resample(data, labels)

    return resampled_data, resampled_labels


def OverSVMSMOTE(request, data, label):
    # 使用 Counter 获取每个唯一值的计数
    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    label = label.ravel()

    # 从表单中获取可调参数
    # n_neighbors <= n_samples
    k_neighbors = int(request.form.get("SVMSMOTE_k", 3))
    out_step = float(request.form.get("SVMSMOTE_out_step", 0.5))
    svm_estimator = None  # 默认SVM分类器

    sampling_strategy = request.form.get("SVMSMOTE_sampling_strategy", "auto")
    # 调用函数进行过采样
    data, label = svm_smote_oversample(data, label, sampling_strategy,
                                       k_neighbors=k_neighbors, svm_estimator=svm_estimator, out_step=out_step)

    # 使用 Counter 获取每个唯一值的计数
    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Now counts:", label_count_dict)

    return data, label


if __name__ == '__main__':

    data_class_0 = np.random.normal(loc=1, scale=0.2, size=(10, 10))
    data_class_1 = np.random.normal(loc=2, scale=0.2, size=(5, 10))
    data_class_2 = np.random.normal(loc=3, scale=0.2, size=(4, 10))

    # 将数据和标签合并
    data = np.vstack((data_class_0, data_class_1, data_class_2))
    labels = np.array([0] * 10 + [1] * 5 + [2] * 4)

    # 设定采样策略
    sampling_strategy = "auto"

    # 调用 SVMSMOTE 函数进行过采样
    resampled_data, resampled_labels = svm_smote_oversample(
        data, labels, sampling_strategy, k_neighbors=3, out_step=0.5
    )

    print("Resampled labels:", Counter(resampled_labels))
