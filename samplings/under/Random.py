from collections import Counter
from sklearn.utils import resample
import numpy as np
import pandas as pd
import os


def UnderRandom(request, data, label):
    # 使用 Counter 获取每个唯一值的计数
    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    for l, n in label_counts.items():
        num = int(request.form.get(f'UnderRandom{l}'))
        label_count_dict[l] = num
    print(label_count_dict)

    data, label = random_downsample(data, label, label_count_dict)

    return data, label


def random_downsample(X, y, sampling_strategy):
    """
    根据给定的采样策略对数据集进行随机欠采样。

    参数:
    X : np.ndarray, 特征矩阵
    y : np.ndarray, 标签数组
    sampling_strategy : dict, 指定每个类的目标样本数量，格式为 {class_label: target_count}

    返回:
    X_resampled : np.ndarray, 欠采样后的特征矩阵
    y_resampled : np.ndarray, 欠采样后的标签数组
    """
    y = y.ravel()
    # 初始化欠采样后的特征和标签数组
    X_resampled = []
    y_resampled = []

    # 对每个类标签进行处理
    unique_classes = np.unique(y)
    for cls in unique_classes:
        # 获取当前类的样本
        X_cls = X[y == cls]
        y_cls = y[y == cls]

        # 如果采样策略中有指定目标数量，则进行欠采样
        if cls in sampling_strategy:
            target_count = sampling_strategy[cls]

            # 欠采样当前类到目标数量
            X_cls_downsampled, y_cls_downsampled = resample(
                X_cls, y_cls,
                replace=False,  # 不允许重复采样
                n_samples=target_count,  # 指定样本数量
                random_state=42
            )

            # 添加欠采样后的数据到结果列表
            X_resampled.append(X_cls_downsampled)
            y_resampled.append(y_cls_downsampled)
        else:
            # 如果没有在sampling_strategy中指定的类，保持其原始数量
            X_resampled.append(X_cls)
            y_resampled.append(y_cls)

    # 合并所有欠采样的类别数据
    X_resampled = np.vstack(X_resampled)
    y_resampled = np.hstack(y_resampled)

    return X_resampled, y_resampled