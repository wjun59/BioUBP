import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection


def calculate_ac_feature(sequence, df, pcplist, lag):
    """
    计算单个序列的AC特征。

    参数:
    - sequence: str, 蛋白质序列。
    - df: DataFrame, 包含理化性质的数据表 (Table 5)。
    - pcplist: list, 选择的理化性质索引（从1开始）。
    - lag: int, 滞后距离。

    返回:
    - ac_features: list, 该序列的AC特征向量。
    """
    # 将pcplist从1基准转换为0基准
    pcplist = [x - 1 for x in pcplist]

    # 获取序列长度
    L = len(sequence)

    # 提取理化性质数据，只保留对应氨基酸列的数据（去掉理化性质名称列）
    property_values = df.iloc[pcplist, 1:].values  # 选取pcplist行，从第2列到第21列

    # 创建字典，将氨基酸残基映射到理化性质值向量
    amino_acids = list("ARNDCQEGHILKMFPSTWYV")
    aa_properties = {aa: property_values[:, i] for i, aa in enumerate(amino_acids)}

    # 计算AC特征
    ac_features = []
    for prop_idx, property_data in enumerate(property_values):
        # 将序列中的氨基酸转换为对应的理化性质值
        seq_values = np.array([aa_properties[aa][prop_idx] for aa in sequence])

        # 计算该理化性质的均值
        mean_value = np.mean(seq_values)

        # 计算自协方差 (AC) 特征
        for k in range(1, lag + 1):
            ac_value = np.mean((seq_values[:L - k] - mean_value) * (seq_values[k:] - mean_value))
            ac_features.append(ac_value)

    return ac_features


def extract_ac_features(sequences, df, pcplist, lag):
    """
    为多条序列提取AC特征。

    参数:
    - sequences: list, 包含多个蛋白质序列的列表。
    - df: DataFrame, 包含理化性质的数据表 (Table 5)。
    - pcplist: list, 选择的理化性质索引（从1开始）。
    - lag: int, 滞后距离。

    返回:
    - features: list of list, 每个序列的AC特征向量组成的列表。
    """
    features = []
    for sequence in sequences:
        ac_features = calculate_ac_feature(sequence, df, pcplist, lag)
        features.append(ac_features)
    return np.array(features)


def pro_ac(seq, request):
    lag = int(request.form.get('ac_lag'))
    dimensions = int(request.form.get('ac_Dimensions'))
    pcplist = request.form.getlist('ac_param3')
    pcplist = [int(x) for x in pcplist]
    df = pd.read_excel("feature_extraction/pychemdata/Table 7.xlsx", sheet_name='Sheet11')
    data = extract_ac_features(seq, df, pcplist, lag)

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
    pcplist = [1, 2, 3, 10, 22, 547]

    # 提取AC特征
    features = extract_ac_features(sequences, df, pcplist, lag)

    # 打印结果
    for i, feature_vector in enumerate(features):
        print(f"Sequence {i + 1} AC features: {feature_vector}")
    print(features.shape)