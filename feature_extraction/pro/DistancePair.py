
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

# （根据提供的分组信息）
clusters_cp13 = [
    {'M', 'F'}, {'I', 'L'}, {'V'}, {'A'}, {'C'}, {'W', 'Y', 'Q', 'H', 'P'},
    {'G'}, {'T'}, {'S'}, {'N'}, {'R', 'K'}, {'D'}, {'E'}]
clusters_cp14 = [
    {'I', 'M', 'V'},
    {'L'},
    {'F'},
    {'W', 'Y'},
    {'G'},
    {'P'},
    {'C'},
    {'A'},
    {'S'},
    {'T'},
    {'N'},
    {'H', 'R', 'K', 'Q'},
    {'E'},
    {'D'}]
clusters_cp19 = [
    {'A'},  # Alanine
    {'R'},  # Arginine
    {'N'},  # Asparagine
    {'D'},  # Aspartic acid
    {'C'},  # Cysteine
    {'Q'},  # Glutamine
    {'E'},  # Glutamic acid
    {'G'},  # Glycine
    {'H'},  # Histidine
    {'I'},  # Isoleucine
    {'L'},  # Leucine
    {'K'},  # Lysine
    {'M'},  # Methionine
    {'F', 'Y'},
    {'P'},  # Proline
    {'S'},  # Serine
    {'T'},  # Threonine
    {'W'},  # Tryptophan
    {'V'},  # Valine
]
clusters_cp20 = [
    {'A'},  # Alanine
    {'R'},  # Arginine
    {'N'},  # Asparagine
    {'D'},  # Aspartic acid
    {'C'},  # Cysteine
    {'Q'},  # Glutamine
    {'E'},  # Glutamic acid
    {'G'},  # Glycine
    {'H'},  # Histidine
    {'I'},  # Isoleucine
    {'L'},  # Leucine
    {'K'},  # Lysine
    {'M'},  # Methionine
    {'F'},  # Phenylalanine
    {'P'},  # Proline
    {'S'},  # Serine
    {'T'},  # Threonine
    {'W'},  # Tryptophan
    {'Y'},  # Tyrosine
    {'V'} ]


def reduced_alphabet_pseaac(sequences, d_max, clusters):
    """
    PseAAC of Distance-Pairs and Reduced Alphabet Scheme

    Parameters:
        sequences: List[str] - 蛋白质序列列表
        d_max: int - 最大距离阈值
        clusters: List[set] - 氨基酸的精简字母表（每个集合表示一组）

    Returns:
        np.ndarray - 特征矩阵
    """
    # 构建氨基酸到精简类别的映射
    aa_to_cluster = {}
    for cluster_idx, cluster in enumerate(clusters):
        for aa in cluster:
            aa_to_cluster[aa] = cluster_idx

    n = len(clusters)  # 精简字母表的类别数量
    feature_dim = n + d_max * n ** 2  # 特征向量的维度
    features = []

    for seq in sequences:
        # 初始化特征向量
        feature_vector = np.zeros(feature_dim)

        # 单类别频率统计
        single_counts = Counter([aa_to_cluster[aa] for aa in seq if aa in aa_to_cluster])
        for cluster_idx, count in single_counts.items():
            feature_vector[cluster_idx] = count

        # 残基对频率统计（基于距离）
        for i in range(len(seq)):
            for d in range(1, d_max + 1):  # 遍历距离
                j = i + d
                if j < len(seq) and seq[i] in aa_to_cluster and seq[j] in aa_to_cluster:
                    cluster_i = aa_to_cluster[seq[i]]
                    cluster_j = aa_to_cluster[seq[j]]
                    pair_index = n + (cluster_i * n + cluster_j) * d_max + (d - 1)
                    feature_vector[pair_index] += 1

        # 归一化
        feature_vector /= np.sum(feature_vector)
        features.append(feature_vector)

    return np.array(features)


def pro_DistancePair(seq, request):
    dimensions = int(request.form.get('distance_Dimensions'))
    d_max = int(request.form.get('distance_d'))
    num = int(request.form.get('distance_clusters'))
    if num == 1:
        clusters = clusters_cp13
    elif num == 2:
        clusters = clusters_cp14
    elif num == 3:
        clusters = clusters_cp19
    else:
        clusters = clusters_cp20

    data = reduced_alphabet_pseaac(seq, d_max, clusters)
    if data.shape[1] > dimensions:
        if dimensions < data.shape[0]:
            pca = PCA(n_components=dimensions)
            data = pca.fit_transform(data)

    if data.shape[1] > dimensions:
        rp = GaussianRandomProjection(n_components=dimensions)
        data = rp.fit_transform(data)
    return data


if __name__ == '__main__':

    # 测试代码
    sequences = ["ACDEFGHIKLMNPQRSTVWY", "AACCGGTTAA"]
    d_max = 5  # 最大距离
    features = reduced_alphabet_pseaac(sequences, d_max, clusters_cp20)

    # 输出特征矩阵
    print("Feature matrix shape:", features.shape)
    print(features)
