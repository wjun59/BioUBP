import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


def rna_revkmer(seq, request):
    k = int(request.form.get('revkmer_k'))
    dimensions = int(request.form.get('revkmer_Dimensions'))
    data = revkmer_features(seq, k)

    if data.shape[1] > dimensions:
        if dimensions < data.shape[0]:
            pca = PCA(n_components=dimensions)
            data = pca.fit_transform(data)
    if data.shape[1] > dimensions:
        rp = GaussianRandomProjection(n_components=dimensions)
        data = rp.fit_transform(data)
    return data


def reverse_complement(kmer):
    """返回kmer的反向互补"""
    complement = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[nuc] for nuc in reversed(kmer))


def get_rev_kmer_features(k):
    """根据k值生成RevKmer特征的标准列表"""
    if k == 1:
        return ['A', 'C', 'G', 'U']
    elif k == 2:
        return ['AA', 'AC', 'AG', 'AU', 'CA', 'CC', 'CG', 'GA', 'GC', 'UA']
    else:
        raise ValueError("Currently only k=1 and k=2 are supported.")


def revkmer_features(sequences, k):
    """提取反向互补kmer特征并生成特征矩阵"""
    # 获取RevKmer特征列表
    rev_kmer_list = get_rev_kmer_features(k)
    feature_count = len(rev_kmer_list)

    # 初始化特征矩阵
    features_matrix = np.zeros((len(sequences), feature_count), dtype=int)

    # 创建特征索引字典
    feature_index = {kmer: idx for idx, kmer in enumerate(rev_kmer_list)}

    # 遍历每个序列，提取特征
    for seq_idx, seq in enumerate(sequences):
        all_kmers = defaultdict(int)

        # 提取kmer并计算出现次数
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            rev_kmer = reverse_complement(kmer)

            # 合并kmer和反向互补kmer
            if kmer < rev_kmer:
                all_kmers[kmer] += 1
            else:
                all_kmers[rev_kmer] += 1

        # 将出现次数填入特征矩阵
        for kmer, count in all_kmers.items():
            if kmer in feature_index:
                features_matrix[seq_idx, feature_index[kmer]] = count

    return np.array(features_matrix)

