import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection

# 定义氨基酸的理化性质
hydrophobicity = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

hydrophilicity = {
    'A': -0.5, 'R': 3, 'N': 1, 'D': 3, 'C': 0,
    'Q': 3, 'E': 2, 'G': 0, 'H': 0.5, 'I': -0.5,
    'L': -0.5, 'K': 3, 'M': 0, 'F': -0.2, 'P': 0,
    'S': 0.6, 'T': 0.5, 'W': 0.1, 'Y': 0.3, 'V': -0.5
}


# 标准化理化属性
def standardize_property(property_dict):
    values = np.array(list(property_dict.values()))
    mean = np.mean(values)
    std = np.std(values)
    return {k: (v - mean) / std for k, v in property_dict.items()}


# 计算氨基酸频率
def calculate_frequency(sequence):
    amino_acids = 'ARNDCQEGHILKMFPSTWYV'
    freq = {aa: sequence.count(aa) / len(sequence) for aa in amino_acids}
    return np.array([freq[aa] for aa in amino_acids])


# 计算序列相关因子 τ_j
def calculate_tau(sequence, properties, max_lambda):
    taus = []
    for j in range(1, max_lambda + 1):
        tau_j = 0
        for i in range(len(sequence) - j):
            amino_acid_1 = sequence[i]
            amino_acid_2 = sequence[i + j]
            product_sum = np.prod(
                [properties[prop][amino_acid_1] * properties[prop][amino_acid_2] for prop in properties])
            tau_j += product_sum
        taus.append(tau_j / (len(sequence) - j))
    return np.array(taus)


# SC-PseAAC 特征提取
def sc_pseaac(sequences, properties_indices, max_lambda=5, weight=0.05):
    properties_all = {
        'hydrophobicity': standardize_property(hydrophobicity),
        'hydrophilicity': standardize_property(hydrophilicity)
    }

    # 根据传入的 indices 选择理化性质
    selected_properties = {}
    property_keys = list(properties_all.keys())
    for idx in properties_indices:
        selected_properties[property_keys[idx - 1]] = properties_all[property_keys[idx - 1]]

    feature_matrix = []
    for sequence in sequences:
        freq_vector = calculate_frequency(sequence)
        tau_vector = calculate_tau(sequence, selected_properties, max_lambda)

        # 构建特征向量
        freq_part = freq_vector / (1 + weight * tau_vector.sum())
        tau_part = (weight * tau_vector) / (1 + weight * tau_vector.sum())
        feature_vector = np.concatenate([freq_part, tau_part])

        feature_matrix.append(feature_vector)

    return np.array(feature_matrix)


def pro_SC_PseAAC(seq, request):
    lambda_val = int(request.form.get('SCPseAAC_lambda'))
    weight = float(request.form.get('SCPseAAC_Weight'))
    dimensions = int(request.form.get('SCPseAAC_Dimensions'))
    pcplist = request.form.getlist('SCPseAAC_param3')
    pcplist = [int(x) for x in pcplist]

    data = sc_pseaac(seq, pcplist, lambda_val, weight)

    if data.shape[1] > dimensions:
        if dimensions < data.shape[0]:
            pca = PCA(n_components=dimensions)
            data = pca.fit_transform(data)

    if data.shape[1] > dimensions:
        rp = GaussianRandomProjection(n_components=dimensions)
        data = rp.fit_transform(data)

    return data


# 测试用例
if __name__ == "__main__":
    sequences = ["VYRNRTLQKWH", "TVPNDYMTSPA", "EDEDVSKEYGH"]
    properties_indices = [1, 2]  # 使用疏水性和亲水性两个属性
    max_lambda = 3  # 设置最大层次
    weight = 0.1  # 设置权重因子

    features = sc_pseaac(sequences, properties_indices, max_lambda, weight)
    print("SC-PseAAC features:")
    print(features)
