import numpy as np

import numpy as np

def gauss_mf(x, c, sigma):
    """高斯隶属度函数"""
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)


def fuzzy_undersampling(data, labels, alpha):
    # 1. 计算正负样本的均值和标准差
    pos_samples = data[labels == 1]
    neg_samples = data[labels == 0]

    # 计算正样本均值和标准差
    pos_mean = np.mean(pos_samples, axis=0)
    pos_std = np.std(pos_samples, axis=0)

    # 计算负样本均值和标准差
    neg_mean = np.mean(neg_samples, axis=0)
    neg_std = np.std(neg_samples, axis=0)

    # 2. 使用高斯隶属度函数计算每个样本的隶属度
    pos_membership = np.exp(-0.5 * ((pos_samples - pos_mean) / pos_std) ** 2)
    neg_membership = np.exp(-0.5 * ((neg_samples - neg_mean) / neg_std) ** 2)

    # 3. 计算每个负样本的得分
    neg_scores = np.sum(neg_membership, axis=1)

    # 4. 计算需要删除的负样本数量
    pos_num = len(pos_samples)
    neg_num = len(neg_samples)
    neg_remove = int(neg_num - (1 - alpha) / alpha * pos_num)

    # 5. 按得分排序，选择最小得分的负样本
    neg_sorted_indices = np.argsort(neg_scores)
    neg_samples_to_keep = neg_samples[neg_sorted_indices[:-neg_remove]]  # 去掉得分最高的负样本

    # 6. 返回删除负样本后的数据
    return np.vstack([pos_samples, neg_samples_to_keep]), np.concatenate([np.ones(pos_num), np.zeros(len(neg_samples_to_keep))])


def UnderFuzzy(request, X, label):
    a = int(request.form.get(f'Fuzzy_alpha'))
    # 希望减少哪个类别周围的样本
    y = label.ravel()
    X_resampled, y_resampled = fuzzy_undersampling(X, y, a)
    return X_resampled, y_resampled


if __name__ == '__main__':
    # 模拟数据
    data = np.random.randn(200, 3)  # 200个样本，3个特征
    labels = np.array([1] * 80 + [0] * 120)  # 80个正样本，120个负样本

    # 设置alpha
    alpha = 0.1
    # α 控制着正负样本的比例，较大值保留更多的负样本，而较小的值则会进行更强的欠采样，删除更多的负样本。

    # 使用模糊欠采样算法
    resampled_data, resampled_labels = fuzzy_undersampling(data, labels, alpha)

    # Output resampled dataset size and class distribution
    print("Resampled dataset size:", resampled_data.shape)
    print("Resampled dataset class distribution:", dict(zip(*np.unique(resampled_labels, return_counts=True))))