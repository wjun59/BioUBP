from imblearn.under_sampling import TomekLinks
from collections import Counter
import numpy as np


def UnderTomekLinks(request, data, labels):
    """
    'auto'（默认）：只从多数类中删除与少数类构成 Tomek链 的样本。
    'all'：清理所有类别之间的边界样本，无论是多数类还是少数类。
    'majority'：清理多数类的样本。
    'not minority'：清理除少数类之外的所有类别的样本。
    'not majority'：清理除多数类之外的所有类别的样本。
    """

    sampling_strategy = request.form.get(f'TomekLinks_sampling_strategy')
    label_counts = Counter(labels.flatten())
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)
    label = labels.ravel()

    resampled_data, resampled_labels = tomek_links_resample(data, label, sampling_strategy)
    label_counts2 = Counter(resampled_labels.flatten())
    label_count_dict2 = dict(label_counts2)
    print("Now counts:", label_count_dict2)

    return resampled_data, resampled_labels


def tomek_links_resample(data, labels, sampling_strategy):
    tl = TomekLinks(sampling_strategy=sampling_strategy)
    resampled_data, resampled_labels = tl.fit_resample(data, labels)
    return resampled_data, resampled_labels


if __name__ == '__main__':
    # 生成每个类别的数据
    X = np.random.rand(100, 4)  # 类别 0 的样本
    y = np.array([0] * 10 + [1] * 20 + [2] * 70)
    sampling_strategy = 'majority'

    x_resampled, y_resampled = tomek_links_resample(X, y, sampling_strategy)
    print(x_resampled.shape)

    unique, counts = np.unique(y_resampled, return_counts=True)
    sampled_distribution = dict(zip(unique, counts))

    print("采样后的类别分布:")
    for label, count in sampled_distribution.items():
        print(f"类别 {label}: {count} 个样本")
