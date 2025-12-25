import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


# 标准化理化性质
def standardize_physicochemical_values(df, pcplist):
    """
    标准化选定理化性质值（从 Table 7 提供的值）。
    Args:
        df (pd.DataFrame): Table 7 数据（20列为氨基酸属性值，第1列为属性名称）。
        pcplist (list): 用户选择的理化性质索引（从1开始）。

    Returns:
        standardized_values (dict): 标准化后的每个选定理化性质的值字典。
    """
    standardized_values = {}
    for idx in pcplist:
        prop_values = df.iloc[idx - 1, 1:21].values.astype(float)  # 提取理化性质值
        mean_val = np.mean(prop_values)
        std_dev = np.std(prop_values)
        standardized_values[idx] = (prop_values - mean_val) / std_dev
    return standardized_values


# 计算相关因子θ
def compute_theta(sequence, lambda_val, standardized_values, pcplist):
    """
    计算序列的相关因子 (θ)。
    Args:
        sequence (str): 蛋白质序列。
        lambda_val (int): 最大延迟等级。
        standardized_values (dict): 标准化理化性质值。
        pcplist (list): 用户选择的理化性质索引。

    Returns:
        theta (np.ndarray): θ 值数组。
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    L = len(sequence)
    theta = np.zeros(lambda_val)

    for d in range(1, lambda_val + 1):
        correlation_sum = 0
        for i in range(L - d):
            rij_sum = 0
            for idx in pcplist:
                Hu_i = standardized_values[idx][amino_acids.index(sequence[i])]
                Hu_j = standardized_values[idx][amino_acids.index(sequence[i + d])]
                rij_sum += (Hu_i - Hu_j) ** 2
            correlation_sum += rij_sum / len(pcplist)
        theta[d - 1] = correlation_sum / (L - d)

    return theta


# 提取单个序列的 PC-PseAAC-General 特征向量
def compute_pc_pseaac_general(sequence, lambda_val, weight, standardized_values, pcplist):
    """
    提取单个蛋白质序列的 PC-PseAAC-General 特征向量。
    Args:
        sequence (str): 蛋白质序列。
        lambda_val (int): 最大延迟等级。
        weight (float): 权重因子。
        standardized_values (dict): 标准化理化性质值。
        pcplist (list): 用户选择的理化性质索引。

    Returns:
        feature_vector (np.ndarray): 序列的特征向量。
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    L = len(sequence)

    # 计算标准化频率 f
    f_normalized = np.array([sequence.count(aa) / L for aa in amino_acids])

    # 计算 θ 值
    theta = compute_theta(sequence, lambda_val, standardized_values, pcplist)

    # 构造特征向量
    denom = 1 + weight * np.sum(theta)
    feature_vector = np.zeros(20 + lambda_val)

    # 前20个分量：氨基酸频率
    feature_vector[:20] = f_normalized / denom

    # 后 lambda_val 个分量：延迟相关因子
    feature_vector[20:] = (weight * theta) / denom

    return feature_vector


# 提取多个序列的特征向量
def extract_pc_pseaac_general(sequences, lambda_val, weight, pcplist, df):
    """
    提取多个蛋白质序列的 PC-PseAAC-General 特征向量。
    Args:
        sequences (list): 蛋白质序列列表。
        lambda_val (int): 最大延迟等级。
        weight (float): 权重因子。
        pcplist (list): 用户选择的理化性质索引。
        df (pd.DataFrame): Table 7 数据。

    Returns:
        feature_matrix (pd.DataFrame): 包含所有序列特征向量的 DataFrame。
    """
    # 标准化理化性质
    standardized_values = standardize_physicochemical_values(df, pcplist)

    # 计算每个序列的特征向量
    feature_matrix = []
    for seq in sequences:
        features = compute_pc_pseaac_general(seq, lambda_val, weight, standardized_values, pcplist)
        feature_matrix.append(features)

    return np.array(feature_matrix)


def pro_PC_PseAACG(seq, request):
    lambda_val = int(request.form.get('PCPseAACGeneral_lambda'))
    weight = float(request.form.get('PCPseAACGeneral_Weight'))
    dimensions = int(request.form.get('PCPseAACGeneral_Dimensions'))
    pcplist = request.form.getlist('PCPseAACGeneral_param3')
    pcplist = [int(x) for x in pcplist]
    df = pd.read_excel("feature_extraction/pychemdata/Table 7.xlsx", sheet_name='Sheet11')
    data = extract_pc_pseaac_general(seq, lambda_val, weight, pcplist, df)

    if data.shape[1] > dimensions:
        if dimensions < data.shape[0]:
            pca = PCA(n_components=dimensions)
            data = pca.fit_transform(data)

    if data.shape[1] > dimensions:
        rp = GaussianRandomProjection(n_components=dimensions)
        data = rp.fit_transform(data)

    return data


# 主函数测试用例
if __name__ == "__main__":
    # 读取 Table 7 数据
    df = pd.read_excel("../pychemdata/Table 7.xlsx", sheet_name='Sheet11')
    # 参数设置
    sequences = ["VYRNRTLQKWH", "TVPNDYMTSPA", "EDEDVSKEYGH"]  # 样本序列
    lambda_val = 5  # 延迟等级
    weight = 0.2  # 权重因子
    pcplist = [1, 2, 3, 547]  # 用户选择的理化性质索引

    # 提取特征
    features_df = extract_pc_pseaac_general(sequences, lambda_val, weight, pcplist, df)
    print(features_df)
