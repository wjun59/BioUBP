import numpy as np
from imblearn.over_sampling import SMOTE


def OverSmote(request, X, y):

    k = int(request.form.get(f'smote_k'))  # 1-10

    data, label = Over_Smote(X=X, y=y, k_neighbors=k)
    return data, label


def Over_Smote(X, y, sampling_strategy='auto', k_neighbors=5, random_state=None):
    """
    对不平衡数据集进行SMOTE过采样。

    参数：
    X : array-like, shape (n_samples, n_features)
        特征数据，包含多个样本的特征信息。

    y : array-like, shape (n_samples,)
        目标标签，与特征数据对应的类别标签。

    k_neighbors : int, default=5
        用于生成合成样本时，选择的少数类样本的邻居数量。

    random_state : int, default=None
        控制随机数生成的种子，以确保结果可重复。

    返回：
    X_resampled : array-like, shape (n_samples_new, n_features)
        经过过采样后的特征数据。

    y_resampled : array-like, shape (n_samples_new,)
        经过过采样后的目标标签。
    """
    # 创建SMOTE对象
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)

    # 执行过采样
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("过采样后的类别分布:", dict(zip(*np.unique(y_resampled, return_counts=True))))
    return X_resampled, y_resampled


# 示例用法
if __name__ == "__main__":
    X = np.random.rand(100, 4)  # 类别 0 的样本
    y = np.array([0] * 10 + [1] * 20 + [2] * 70)

    # 使用OverSmote进行过采样
    X_resampled, y_resampled = Over_Smote(X, y, k_neighbors=5, random_state=42)

    # 检查结果
    print("过采样后的类别分布:", dict(zip(*np.unique(y_resampled, return_counts=True))))
