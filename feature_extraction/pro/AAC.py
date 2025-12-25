from collections import Counter
import numpy as np

# 20种氨基酸的顺序
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


# 计算AAC特征向量
def extract_aac_feature(sequences):
    aac_vectors = []

    for peptide in sequences:
        # 统计每个氨基酸在序列中的频率
        count = Counter(peptide)
        total_count = len(peptide)

        # 防止除零错误
        if total_count == 0:
            # 如果序列为空，AAC向量为全零
            aac_vectors.append([0] * 20)
            continue

        # 计算AAC特征向量
        aac_vector = []
        for aa in amino_acids:
            # 计算每种氨基酸的频率
            aac_vector.append(count.get(aa, 0) / total_count)

        aac_vectors.append(aac_vector)

    return np.array(aac_vectors)


def pro_aac(seq, request):
    data = extract_aac_feature(seq)

    return data
