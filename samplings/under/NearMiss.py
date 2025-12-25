from imblearn.under_sampling import NearMiss
from collections import Counter


def UnderNearMiss(request, X, label):
    version = int(request.form.get(f'NearMiss_version'))
    label_counts = Counter(label.flatten())
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    for l, n in label_counts.items():
        num = int(request.form.get(f'NearMiss{l}'))
        label_count_dict[l] = num
    print(label_count_dict)
    label = label.ravel()
    if version == 1:
        resampled_data, resampled_labels = nearmiss1_resample(X, label, label_count_dict)
    elif version == 2:
        resampled_data, resampled_labels = nearmiss2_resample(X, label, label_count_dict)
    else:
        resampled_data, resampled_labels = nearmiss3_resample(X, label, label_count_dict)
    return resampled_data, resampled_labels


def nearmiss1_resample(data, labels, sampling_strategy):
    nm = NearMiss(version=1, sampling_strategy=sampling_strategy)
    resampled_data, resampled_labels = nm.fit_resample(data, labels)
    return resampled_data, resampled_labels


def nearmiss2_resample(data, labels, sampling_strategy):
    nm = NearMiss(version=2, sampling_strategy=sampling_strategy)
    resampled_data, resampled_labels = nm.fit_resample(data, labels)
    return resampled_data, resampled_labels


def nearmiss3_resample(data, labels, sampling_strategy):
    nm = NearMiss(version=3, sampling_strategy=sampling_strategy)
    resampled_data, resampled_labels = nm.fit_resample(data, labels)
    return resampled_data, resampled_labels