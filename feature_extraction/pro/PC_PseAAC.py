import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection
# 给定理化性质数据
properties = {
    "hydrophobicity": {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
        'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    },
    "hydrophilicity": {
        'A': -0.5, 'R': 3, 'N': 1, 'D': 3, 'C': 0, 'Q': 3, 'E': 2,
        'G': 0, 'H': 0.5, 'I': -0.5, 'L': -0.5, 'K': 3, 'M': 0, 'F': -0.2,
        'P': 0, 'S': 0.6, 'T': 0.5, 'W': 0.1, 'Y': 0.3, 'V': -0.5
    },
    "mass": {
        'A': 89.09, 'R': 174.2, 'N': 132.12, 'D': 133.1, 'C': 121.16, 'Q': 147.13, 'E': 146.15,
        'G': 75.07, 'H': 155.16, 'I': 131.17, 'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19,
        'P': 115.13, 'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
    }
}


# 标准化理化性质
def normalize_property(values):
    mean_val = np.mean(list(values.values()))
    std_val = np.std(list(values.values()))
    return {k: (v - mean_val) / std_val for k, v in values.items()}


# 根据索引选择指定的理化性质
def select_properties(properties, property_indices):
    prop_keys = ["hydrophobicity", "hydrophilicity", "mass"]
    selected_props = {prop_keys[i - 1]: properties[prop_keys[i - 1]] for i in property_indices if
                      1 <= i <= len(prop_keys)}
    return selected_props


# 计算相关因子θ
def calculate_theta(sequence, selected_properties, lag):
    L = len(sequence)
    theta = []
    for d in range(1, lag + 1):
        if d < L:
            sum_theta = 0
            for i in range(L - d):
                Ri, Rj = sequence[i], sequence[i + d]
                sum_theta += sum((selected_properties[prop][Ri] - selected_properties[prop][Rj]) ** 2 for prop in
                                 selected_properties)
            theta.append(sum_theta / (L - d))
        else:
            theta.append(0)
    return theta


# 提取 PC-PseAAC 特征向量 (输入为序列列表)
def extract_pc_pseaac(sequences, properties, property_indices, lambda_val, weight):
    amino_acids = properties["hydrophobicity"].keys()
    feature_vectors = []

    # 根据索引选择指定的理化性质
    selected_properties = select_properties(properties, property_indices)

    for sequence in sequences:
        # 计算氨基酸频率 fi
        L = len(sequence)
        fi = {aa: sequence.count(aa) / L for aa in amino_acids}

        # 计算 θ 值
        theta_values = calculate_theta(sequence, selected_properties, lambda_val)

        # 计算特征向量的前 20 个分量
        denom = 1 + weight * sum(theta_values)
        feature_vector = [fi[aa] / denom for aa in amino_acids]

        # 计算特征向量的后续分量
        feature_vector += [weight * theta / denom for theta in theta_values]

        feature_vectors.append(feature_vector)

    return np.array(feature_vectors)


def pro_PC_PseAAC(seq, request):
    lambda_val = int(request.form.get('PCPseAAC_lambda'))
    weight = float(request.form.get('PCPseAAC_Weight'))
    dimensions = int(request.form.get('PCPseAAC_Dimensions'))
    pcplist = request.form.getlist('PCPseAAC_param3')
    pcplist = [int(x) for x in pcplist]

    normalized_properties = {prop: normalize_property(values) for prop, values in properties.items()}
    data = extract_pc_pseaac(seq, normalized_properties, pcplist, lambda_val, weight)

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
    # 归一化理化性质
    normalized_properties = {prop: normalize_property(values) for prop, values in properties.items()}

    # 参数设置
    lambda_val = 1  # 最高计数等级
    weight = 0.2  # 权重因子
    property_indices = [1, 2, 3]  # 选择所有三个理化性质

    # 序列列表
    sequences = ["VYRNRTLQKWH", "TVPNDYMTSPA", "EDEDVSKEYGH"]

    # 计算所有序列的 PC-PseAAC 特征向量
    feature_vectors = extract_pc_pseaac(sequences, normalized_properties, property_indices, lambda_val, weight)

    print(feature_vectors.shape)

    # 输出特征向量
    for i, fv in enumerate(feature_vectors):
        print(f"Sequence {i + 1} Feature Vector:", fv)
