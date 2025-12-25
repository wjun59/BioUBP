import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection


def calculate_pdt(sequence, df, property_index, max_lag):
    """
    计算蛋白质序列的 PDT 特征
    Args:
        sequence (str): 蛋白质序列
        df (pd.DataFrame): 理化性质数据表格
        property_index (int): 理化性质索引（0基准）
        max_lag (int): 最大滞后距离
    Returns:
        list: 包含该理化性质不同滞后距离的 PDT 特征值
    """
    L = len(sequence)
    property_values = df.iloc[property_index, 1:].values  # 获取理化性质值（跳过名称列）

    # 创建字典，将氨基酸残基映射到理化性质值
    amino_acids = list("ARNDCQEGHILKMFPSTWYV")
    aa_properties = {aa: property_values[i] for i, aa in enumerate(amino_acids)}

    # 获取序列的属性值向量
    seq_values = np.array([aa_properties[aa] for aa in sequence])

    # 计算每个滞后距离的 PDT 特征值
    pdt_features = []
    for d in range(1, max_lag + 1):
        if d < L:
            pdt_value = np.sum(np.abs(seq_values[:L - d] - seq_values[d:]))
            pdt_features.append(pdt_value)
        else:
            pdt_features.append(0)  # 如果滞后距离超过序列长度，则特征值设为0

    return pdt_features


def extract_pdt_features(sequences, df, pcplist, max_lag):
    """
    提取多个蛋白质序列的 PDT 特征
    Args:
        sequences (list): 蛋白质序列列表
        df (pd.DataFrame): 理化性质数据表格
        pcplist (list): 选择的理化性质索引列表（1基准）
        max_lag (int): 最大滞后距离
    Returns:
        list: 包含所有序列的 PDT 特征的列表
    """
    features = []
    # 将 pcplist 从 1 基准转换为 0 基准
    pcplist = [x - 1 for x in pcplist]

    for sequence in sequences:
        pdt_vector = []

        # 对每种理化性质计算 PDT 特征
        for property_index in pcplist:
            pdt_features = calculate_pdt(sequence, df, property_index, max_lag)
            pdt_vector.extend(pdt_features)

        features.append(pdt_vector)

    return np.array(features)


def pro_PDT(seq, request):
    lag = int(request.form.get('PDT_lag'))
    dimensions = int(request.form.get('PDT_Dimensions'))
    df = pd.read_excel("feature_extraction/pychemdata/Table 7.xlsx", sheet_name='Sheet11')
    pcplist = list(range(1, 548))
    data = extract_pdt_features(seq, df, pcplist, lag)

    if data.shape[1] > dimensions:
        if dimensions < data.shape[0]:
            pca = PCA(n_components=dimensions)
            data = pca.fit_transform(data)

    if data.shape[1] > dimensions:
        rp = GaussianRandomProjection(n_components=dimensions)
        data = rp.fit_transform(data)

    return data


if __name__ == '__main__':
    # 加载数据
    # df = pd.read_excel("feature_extraction/pychemdata/Table 7.xlsx", sheet_name='Sheet1')
    df = pd.read_excel("../pychemdata/Table 7.xlsx", sheet_name='Sheet11')
    # 输入参数
    sequences = ["VYRNRTLQKWH", "TVPNDYMTSPA", "EDEDVSKEYGH"]
    lag = 3
    pcplist = list(range(1, 548))  # 使用 Table 5 中所有 547 种理化性质

    # 提取AC特征
    features = extract_pdt_features(sequences, df, pcplist, lag)
    print(features.shape)
    # 打印结果
    for i, feature_vector in enumerate(features):
        print(f"Sequence {i + 1} AC features: {feature_vector}")