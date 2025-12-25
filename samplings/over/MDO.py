from sklearn.metrics import pairwise_distances
import numpy as np
from collections import Counter
'''
简介:
MDO 是一种自适应上采样方法，通过测量少数类样本到多数类样本的距离来调整上采样的策略。
生成的新样本位置不仅基于少数类样本之间的距离，也考虑了多数类的分布。这种方法通过避免
新样本过度靠近多数类来减少类别间的重叠。
'''


def mdo_upsample(data, labels, sampling_strategy):
    resampled_data = []
    resampled_labels = []

    unique_labels = np.unique(labels)
    majority_label = max(unique_labels, key=lambda l: len(labels[labels == l]))

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        label_data = data[label_indices]

        target_count = sampling_strategy.get(label, len(label_data))

        if target_count <= len(label_data):
            # 下采样或保持不变
            sampled_indices = np.random.choice(len(label_data), target_count, replace=False)
            resampled_data.extend(label_data[sampled_indices])
            resampled_labels.extend([label] * target_count)
        else:
            # 上采样：多数类和少数类都需要根据目标数量来调整
            num_samples_to_generate = target_count - len(label_data)

            # 如果是多数类或样本数量不足，直接随机采样
            if label == majority_label or num_samples_to_generate <= 0:
                sampled_indices = np.random.choice(len(label_data), num_samples_to_generate, replace=True)
                resampled_data.extend(label_data[sampled_indices])
                resampled_labels.extend([label] * num_samples_to_generate)
            else:
                # 对少数类进行上采样
                majority_indices = np.where(labels == majority_label)[0]
                majority_data = data[majority_indices]

                # 计算少数类样本到多数类样本的距离
                distances = pairwise_distances(label_data, majority_data)

                for _ in range(num_samples_to_generate):
                    sample_idx = np.random.randint(0, len(label_data))
                    dist_to_majority = distances[sample_idx]
                    closest_majority_idx = np.argmin(dist_to_majority)

                    # 基于最近的多数类点方向生成新样本
                    direction = majority_data[closest_majority_idx] - label_data[sample_idx]
                    new_sample = label_data[sample_idx] + np.random.rand() * direction

                    resampled_data.append(new_sample)
                    resampled_labels.append(label)

            resampled_data.extend(label_data)
            resampled_labels.extend([label] * len(label_data))

    resampled_data = np.array(resampled_data)
    resampled_labels = np.array(resampled_labels)

    return resampled_data, resampled_labels


def OverMDO(request, data, label):
    # 使用 Counter 获取每个唯一值的计数
    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    for l, n in label_counts.items():
        num = int(request.form.get(f'MDO{l}'))
        label_count_dict[l] = num
    print(label_count_dict)
    label = label.ravel()
    data, label = mdo_upsample(data, label, label_count_dict)

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
    resampled_data, resampled_labels = mdo_upsample(
        data, labels, sampling_strategy
    )

    print(resampled_labels)
    print(resampled_data)








