from collections import Counter
import numpy as np

# 定义疏水性和亲水性氨基酸
hydrophobic = ['A', 'C', 'F', 'I', 'L', 'W', 'V', 'M', 'P', 'Y']  # 疏水性氨基酸
hydrophilic = ['S', 'T', 'N', 'Q', 'G', 'H', 'R', 'K', 'D', 'E']  # 亲水性氨基酸


# 计算一个蛋白质序列中的疏水性和亲水性氨基酸的比例
def classify_amino_acids(peptide):
    # 统计肽段中的每个氨基酸的频率
    count = Counter(peptide)

    # 初始化疏水性和亲水性的计数器
    hydrophobic_count = 0
    hydrophilic_count = 0

    # 遍历肽段中的每个氨基酸
    for aa, num in count.items():
        if aa in hydrophobic:
            hydrophobic_count += num
        elif aa in hydrophilic:
            hydrophilic_count += num

    return hydrophobic_count, hydrophilic_count


# 提取蛋白质序列的特征向量（对于每个肽段）
def extract_hhaa_feature(sequences):
    feature_vectors = []

    for peptide in sequences:
        # 计算疏水性和亲水性氨基酸的频率
        hydrophobic_count, hydrophilic_count = classify_amino_acids(peptide)

        # 总氨基酸数
        total_count = len(peptide)

        # 防止除零错误
        if total_count == 0:
            feature_vectors.append([0, 0])
            continue

        # 计算比例，返回2维特征向量
        hydrophobic_ratio = hydrophobic_count / total_count
        hydrophilic_ratio = hydrophilic_count / total_count

        # 将特征向量添加到列表中
        feature_vectors.append([hydrophobic_ratio, hydrophilic_ratio])

    return np.array(feature_vectors)


def pro_chhaa(seq, request):
    data = extract_hhaa_feature(seq)
    return data
