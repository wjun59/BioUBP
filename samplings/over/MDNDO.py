import numpy as np
from collections import Counter

def Over_MDNDO(data, labels, sampling_strategy):
    """
    Perform Gaussian noise sampling on the input data to achieve the target sample sizes
    specified in sampling_strategy, which includes both original and generated data.

    Parameters:
    - data: numpy array, shape (n_samples, n_features), the input data.
    - labels: numpy array, shape (n_samples,), the labels corresponding to the input data.
    - sampling_strategy: dict, specifying the total number of samples (original + generated) for each class.

    Returns:
    - resampled_data: numpy array, the resampled data after Gaussian noise sampling.
    - resampled_labels: numpy array, the labels corresponding to the resampled data.
    """

    def generate_gaussian_samples(sample, num_samples):
        """
        Generate Gaussian noise samples based on the input sample.

        Parameters:
        - sample: numpy array, shape (n_features,), a single input sample.
        - num_samples: int, number of samples to generate.

        Returns:
        - gaussian_samples: numpy array, shape (num_samples, n_features), generated samples.
        """
        cov_matrix = np.diag(0.05 * np.square(sample)).astype('float64')
        sample = sample.astype('float64')
        gaussian_samples = np.random.multivariate_normal(sample, cov_matrix, num_samples)
        return gaussian_samples

    resampled_data = []
    resampled_labels = []

    # Process each class based on the sampling strategy
    for label, total_samples in sampling_strategy.items():
        class_data = data[labels == label]
        original_sample_count = len(class_data)

        # Calculate the number of new samples needed
        new_samples_needed = max(total_samples - original_sample_count, 0)

        # Add the original samples to the resampled data
        resampled_data.append(class_data)
        resampled_labels.extend([label] * original_sample_count)

        # Generate additional samples if needed
        if new_samples_needed > 0:
            generated_samples = []
            while len(generated_samples) < new_samples_needed:
                for sample in class_data:
                    if len(generated_samples) >= new_samples_needed:
                        break
                    gaussian_samples = generate_gaussian_samples(sample, 1)
                    generated_samples.append(gaussian_samples[0])
                    resampled_labels.append(label)
            resampled_data.append(np.array(generated_samples))

    # Combine all generated and original data
    resampled_data = np.vstack(resampled_data)
    resampled_labels = np.array(resampled_labels)

    return resampled_data, resampled_labels


def OverMDNDO(request, data, label):
    # 使用 Counter 获取每个唯一值的计数
    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    for l, n in label_counts.items():
        num = int(request.form.get(f'MDNDO{l}'))
        label_count_dict[l] = num
    print(label_count_dict)

    label = label.ravel()
    data, label = Over_MDNDO(data, label, label_count_dict)

    return data, label


if __name__ == '__main__' :
    np.random.seed(42)
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
    resampled_data, resampled_labels = Over_MDNDO(
        data, labels, sampling_strategy
    )

    print(resampled_labels)