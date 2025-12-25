import numpy as np
from collections import Counter

'''
简介:
插值上采样通过在现有样本之间插值来生成新的样本。这种方法通过在少数类样本之间创建中间点来增加数据多样性，可以更好地覆盖数据空间。

'''

def interpolation_oversample(data, labels, sampling_strategy):
    resampled_data = []
    resampled_labels = []

    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        label_data = data[label_indices]

        target_count = sampling_strategy.get(label, len(label_data))

        if target_count <= len(label_data):
            resampled_data.extend(label_data)
            resampled_labels.extend([label] * len(label_data))
        else:
            # 插值上采样
            num_samples_to_generate = target_count - len(label_data)
            for _ in range(num_samples_to_generate):
                sample_indices = np.random.choice(len(label_data), 2, replace=True)
                new_sample = (label_data[sample_indices[0]] + label_data[sample_indices[1]]) / 2
                resampled_data.append(new_sample)

            resampled_data.extend(label_data)
            resampled_labels.extend([label] * target_count)

    return np.array(resampled_data), np.array(resampled_labels)


def OverInterpolation(request, data, label):
    # 使用 Counter 获取每个唯一值的计数
    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    for l, n in label_counts.items():
        num = int(request.form.get(f'Interpolation{l}'))
        label_count_dict[l] = num
    print(label_count_dict)
    label = label.ravel()
    data, label = interpolation_oversample(data, label, label_count_dict)

    return data, label



if __name__ == '__main__' :
    data_class_0 = np.random.normal(loc=1, scale=0.2, size=(10, 2))
    data_class_1 = np.random.normal(loc=5, scale=0.2, size=(5, 2))
    data_class_2 = np.random.normal(loc=3, scale=0.2, size=(5, 2))

    # 将数据和标签合并
    data = np.vstack((data_class_0, data_class_1))
    data = np.vstack((data, data_class_2))
    labels = np.array([0] * 10 + [1] * 5 + [2]*5)

    # 设定采样策略：将类别 1 扩增到 10 个样本
    sampling_strategy = {0: 10, 1: 10, 2: 10}

    # 调用 kpca_oversample 函数进行过采样
    resampled_data, resampled_labels = interpolation_oversample(
        data, labels, sampling_strategy
    )

    print(resampled_labels)
    print(resampled_data)
