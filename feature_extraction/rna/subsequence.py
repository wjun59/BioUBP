from itertools import product, combinations
import numpy as np

def generate_subsequence_patterns(k):
    """生成所有可能的非连续K-mer模式"""
    kmers = [''.join(p) for p in product('ACGU', repeat=k)]
    subseq_patterns = set()

    # 生成所有可能的子序列模式
    for kmer in kmers:
        for num_gaps in range(k):
            for gap_positions in combinations(range(k), num_gaps):
                pattern = list(kmer)
                for pos in gap_positions:
                    pattern[pos] = '*'
                subseq_patterns.add(''.join(pattern))

    return sorted(subseq_patterns)


def subsequence_feature_vectors(rna_sequences, k, delta):
    # 生成所有可能的非连续K-mer模式
    all_patterns = generate_subsequence_patterns(k)

    # 初始化所有序列的特征向量矩阵
    feature_vectors = []

    # 遍历每个RNA序列
    for rna_sequence in rna_sequences:
        # 初始化当前序列的特征向量
        feature_vector = [0.0] * len(all_patterns)

        # 遍历RNA序列，生成所有K-mer，并统计每个非连续模式的匹配情况
        for i in range(len(rna_sequence) - k + 1):
            kmer = rna_sequence[i:i + k]
            for idx, pattern in enumerate(all_patterns):
                if match_pattern(kmer, pattern):
                    l_value = pattern.count('*')  # 计算非连续匹配的空位数
                    feature_vector[idx] += delta ** l_value

        # 添加到特征向量矩阵中
        feature_vectors.append(feature_vector)

    return np.array(feature_vectors)


def match_pattern(kmer, pattern):
    """判断K-mer是否与给定的模式匹配（考虑非连续匹配）"""
    if len(kmer) != len(pattern):
        return False
    for i in range(len(kmer)):
        if pattern[i] != '*' and kmer[i] != pattern[i]:
            return False
    return True


from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


def rna_subsequence(seq, request):
    k = int(request.form.get('subsequence_k'))
    delta = float(request.form.get('subsequence_delta'))
    dimensions = int(request.form.get('subsequence_Dimensions'))
    data = subsequence_feature_vectors(seq, k, delta)

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
    delta = 0.1  # 衰减因子 δ
    feature_vectors = subsequence_feature_vectors(rna_sequences, k, delta)

    # 打印每个RNA序列的特征向量
    for i, vector in enumerate(feature_vectors):
        print(f"RNA序列 {i + 1} 的特征向量: {vector}")
