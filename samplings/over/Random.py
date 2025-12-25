from collections import Counter
from sklearn.utils import resample
import numpy as np
import pandas as pd
import os


def OverRandom(request, data, label):
    # 使用 Counter 获取每个唯一值的计数
    label_counts = Counter(label.flatten())
    # 转换为字典格式（可选，便于查看）
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    for l, n in label_counts.items():
        num = int(request.form.get(f'OverRandom{l}'))
        label_count_dict[l] = num
    print(label_count_dict)

    data, label = random_upsample(data, label, label_count_dict)

    return data, label


def random_upsample(X, y, sampling_strategy):
    """
    根据给定的采样策略对数据集进行随机上采样。

    参数:
    X : np.ndarray, 特征矩阵
    y : np.ndarray, 标签数组
    sampling_strategy : dict, 指定每个类的目标样本数量，格式为 {class_label: target_count}

    返回:
    X_resampled : np.ndarray, 上采样后的特征矩阵
    y_resampled : np.ndarray, 上采样后的标签数组
    """
    y = y.ravel()
    # 初始化上采样后的特征和标签数组
    X_resampled = []
    y_resampled = []

    # 对每个类标签进行处理
    unique_classes = np.unique(y)
    for cls in unique_classes:
        # 获取当前类的样本
        X_cls = X[y == cls]
        y_cls = y[y == cls]

        # 如果采样策略中有指定目标数量，则进行上采样
        if cls in sampling_strategy:
            target_count = sampling_strategy[cls]

            # 上采样当前类到目标数量
            X_cls_upsampled, y_cls_upsampled = resample(
                X_cls, y_cls,
                replace=True,
                n_samples=target_count,
                random_state=42
            )

            # 添加上采样后的数据到结果列表
            X_resampled.append(X_cls_upsampled)
            y_resampled.append(y_cls_upsampled)
        else:
            # 如果没有在sampling_strategy中指定的类，保持其原始数量
            X_resampled.append(X_cls)
            y_resampled.append(y_cls)

    # 合并所有上采样的类别数据
    X_resampled = np.vstack(X_resampled)
    y_resampled = np.hstack(y_resampled)

    return X_resampled, y_resampled