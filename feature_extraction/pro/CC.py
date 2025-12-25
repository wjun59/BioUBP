import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection
def calculate_cc_feature(sequence, df, u1, u2, lag):
    """
    计算蛋白质序列的交叉协方差（CC）特征
    Args:
        sequence (str): 蛋白质序列
        df (pd.DataFrame): 理化性质数据表格
        u1 (int): 第一种理化性质索引（0基准）
        u2 (int): 第二种理化性质索引（0基准）
        lag (int): 滞后参数
    Returns:
        float: CC 特征值
    """
    # 获取序列长度
    L = len(sequence)

    # 提取理化性质的数值（跳过名称列）
    property_u1 = df.iloc[u1, 1:].values  # 第一个理化性质
    property_u2 = df.iloc[u2, 1:].values  # 第二个理化性质

    # 创建字典，将氨基酸残基映射到理化性质值
    amino_acids = list("ARNDCQEGHILKMFPSTWYV")
    aa_properties_u1 = {aa: property_u1[i] for i, aa in enumerate(amino_acids)}
    aa_properties_u2 = {aa: property_u2[i] for i, aa in enumerate(amino_acids)}

    # 获取序列的属性值向量
    seq_values_u1 = np.array([aa_properties_u1[aa] for aa in sequence])
    seq_values_u2 = np.array([aa_properties_u2[aa] for aa in sequence])

    # 计算两个理化性质的平均值
    mean_u1 = np.mean(seq_values_u1)
    mean_u2 = np.mean(seq_values_u2)

    # 计算交叉协方差特征
    cc_value = np.mean((seq_values_u1[:L - lag] - mean_u1) * (seq_values_u2[lag:] - mean_u2))
    return cc_value


def extract_cc_features(sequences, df, pcplist, lag):
    """
    提取多个蛋白质序列的 CC 特征
    Args:
        sequences (list): 蛋白质序列列表
        df (pd.DataFrame): 理化性质数据表格
        pcplist (list): 选择的理化性质索引列表（1基准）
        lag (int): 最大滞后参数
    Returns:
        list: 包含所有序列的 CC 特征的列表
    """
    features = []
    # 将 pcplist 从 1 基准转换为 0 基准
    pcplist = [x - 1 for x in pcplist]

    for sequence in sequences:
        cc_features = []
        for i in range(len(pcplist)):
            for j in range(i + 1, len(pcplist)):
                for k in range(1, lag + 1):
                    cc_value = calculate_cc_feature(sequence, df, pcplist[i], pcplist[j], k)
                    cc_features.append(cc_value)
        features.append(cc_features)

    return np.array(features)


def pro_cc(seq, request):
    lag = int(request.form.get('cc_lag'))
    dimensions = int(request.form.get('cc_Dimensions'))
    pcplist = request.form.getlist('cc_param3')
    pcplist = [int(x) for x in pcplist]
    df = pd.read_excel("feature_extraction/pychemdata/Table 7.xlsx", sheet_name='Sheet11')
    data = extract_cc_features(seq, df, pcplist, lag)

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
    features = extract_cc_features(sequences, df, pcplist, lag)
    print(features.shape)
    # 打印结果
    for i, feature_vector in enumerate(features):
        print(f"Sequence {i + 1} AC features: {feature_vector}")
