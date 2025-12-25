from itertools import product

import numpy as np


def generate_kmers(k):
    """生成所有可能的K-mer"""
    return [''.join(p) for p in product('ACGU', repeat=k)]


def hamming_distance(s1, s2):
    """计算两个字符串之间的汉明距离"""
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))


def mismatch_feature_vector(rna_sequences, k, m):
    # 生成所有可能的K-mer
    all_kmers = generate_kmers(k)

    # 初始化所有序列的特征向量矩阵
    feature_vectors = []

    # 遍历每个RNA序列
    for rna_sequence in rna_sequences:
        # 初始化当前序列的特征向量
        feature_vector = [0] * len(all_kmers)

        # 遍历RNA序列，生成所有K-mer，并统计每个K-mer的错配情况
        for i in range(len(rna_sequence) - k + 1):
            kmer = rna_sequence[i:i + k]
            for idx, possible_kmer in enumerate(all_kmers):
                if hamming_distance(kmer, possible_kmer) <= m:
                    feature_vector[idx] += 1

        # 添加到特征向量矩阵中
        feature_vectors.append(feature_vector)

    return np.array(feature_vectors)


from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


def rna_mismatch(seq, request):
    k = int(request.form.get('mismatch_k'))
    dimensions = int(request.form.get('mismatch_Dimensions'))
    m = request.form.getlist('mismatch_m')

    data = mismatch_feature_vector(seq, k, m)

    if data.shape[1] > dimensions:
        if dimensions < data.shape[0]:
            pca = PCA(n_components=dimensions)
            data = pca.fit_transform(data)
    if data.shape[1] > dimensions:
        rp = GaussianRandomProjection(n_components=dimensions)
        data = rp.fit_transform(data)
    return data


if __name__ == '__main__':

    # 示例用法
    rna_sequences = ["AUGCAGUCAUACG", "CUGCAGUCAUAG", "AUGCGAUCAUACG"]
    k = 3
    m = 1
    feature_vectors = mismatch_feature_vector(rna_sequences, k, m)

    # 打印特征向量矩阵
    for i, vector in enumerate(feature_vectors):
        print(f"RNA序列 {i + 1} 的特征向量: {vector}")
