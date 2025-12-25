import itertools
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


def rna_kmer(seq, request):
    k = int(request.form.get('kmer_k'))
    dimensions = int(request.form.get('kmer_Dimensions'))
    data = kmer_feature_extraction(seq, k)

    if data.shape[1] > dimensions:
        if dimensions < data.shape[0]:
            pca = PCA(n_components=dimensions)
            data = pca.fit_transform(data)
    if data.shape[1] > dimensions:
        rp = GaussianRandomProjection(n_components=dimensions)
        data = rp.fit_transform(data)
    return data


def kmer_feature_extraction(sequences, k):
    # 获取所有可能的 k-mer 组合
    def generate_kmers(k):
        bases = ['A', 'U', 'C', 'G']
        return [''.join(p) for p in itertools.product(bases, repeat=k)]

    # 所有可能的 k-mer 组合列表
    possible_kmers = generate_kmers(k)
    kmer_index = {kmer: idx for idx, kmer in enumerate(possible_kmers)}
    n_kmers = len(possible_kmers)

    all_features = []
    for seq in sequences:
        # 提取序列中的 k-mers 频率
        kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
        kmer_counts = Counter(kmers)

        # 将频率存储为向量
        kmer_vector = np.zeros(n_kmers)
        for kmer, count in kmer_counts.items():
            if kmer in kmer_index:  # 确保 k-mer 是有效的
                kmer_vector[kmer_index[kmer]] = count

        # 归一化以得到相对频率
        kmer_vector /= np.sum(kmer_vector)
        all_features.append(kmer_vector)

    return np.array(all_features)
