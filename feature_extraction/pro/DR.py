import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection


def distance_based_residue(sequences, d_max):
    # 20种标准氨基酸
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_index = {aa: idx for idx, aa in enumerate(amino_acids)}

    # 特征维度：20 + 20 * 20 * d_max
    feature_dim = 20 + 20 * 20 * d_max
    features = []

    for seq in sequences:
        # 初始化特征向量
        feature_vector = np.zeros(feature_dim)

        # 单个氨基酸频率统计
        single_counts = Counter(seq)
        for aa, count in single_counts.items():
            if aa in aa_index:
                feature_vector[aa_index[aa]] = count

        # 两个氨基酸对频率统计（距离相关）
        for i in range(len(seq)):
            for d in range(1, d_max + 1):  # 遍历距离
                j = i + d
                if j < len(seq) and seq[i] in aa_index and seq[j] in aa_index:
                    idx_i = aa_index[seq[i]]
                    idx_j = aa_index[seq[j]]
                    pair_index = 20 + (idx_i * 20 + idx_j) * d_max + (d - 1)
                    feature_vector[pair_index] += 1

        # 归一化
        feature_vector /= np.sum(feature_vector)
        features.append(feature_vector)

    return np.array(features)


def pro_dr(seq, request):

    dimensions = int(request.form.get('dr_Dimensions'))
    d_max = int(request.form.get('dr_d'))
    data = distance_based_residue(seq, d_max)

    if data.shape[1] > dimensions:
        if dimensions < data.shape[0]:
            pca = PCA(n_components=dimensions)
            data = pca.fit_transform(data)

    if data.shape[1] > dimensions:
        rp = GaussianRandomProjection(n_components=dimensions)
        data = rp.fit_transform(data)

    return data


if __name__ == '__main__':
    # 样例蛋白质序列
    sequences = ["ACDEFGHIKLMNPQRSTVWY", "AACCGGTTAA"]
    d_max = 2  # 最大距离
    features = distance_based_residue(sequences, d_max)

    # 输出特征矩阵
    print("Feature matrix shape:", features.shape)
    print(features)


