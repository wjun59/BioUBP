import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

def pc_psednc_general(rna_sequences, pyche_df, pyche_list, lambda_value, weight):
    """
    计算多个RNA序列的PC-PseDNC-General特征

    参数：
    rna_sequences: list，RNA序列的数组
    pyche_df: pandas.DataFrame，包含理化性质的DataFrame（列为二核苷酸，行为理化性质）
    pyche_list: list，选择用于计算的理化性质的索引值列表
    lambda_value: int，最高相关性阶数（λ）
    weight: float，权重因子（0-1之间）

    返回：
    list，每个RNA序列的PC-PseDNC-General特征向量
    """
    all_feature_vectors = []

    # 遍历每个RNA序列
    for rna_sequence in rna_sequences:
        # 计算序列长度
        L = len(rna_sequence)

        # 确保序列长度足够大以计算给定的lambda
        if L <= lambda_value + 2:
            raise ValueError("RNA序列长度必须大于lambda值加2")

        # 计算二核苷酸的频率（fk）
        dinucleotides = [rna_sequence[i:i + 2] for i in range(L - 1)]
        dinucleotide_freq = {dinucleotide: dinucleotides.count(dinucleotide) / (L - 1)
                             for dinucleotide in pyche_df.columns}

        # 计算theta序列顺序相关因子
        theta = []
        for j in range(1, lambda_value + 1):
            theta_j = 0
            for i in range(L - j - 1):
                dinucleotide_1 = rna_sequence[i:i + 2]
                dinucleotide_2 = rna_sequence[i + j:i + j + 2]
                if dinucleotide_1 in pyche_df.columns and dinucleotide_2 in pyche_df.columns:
                    sum_property_diff = 0
                    for idx in pyche_list:
                        property_name = pyche_df.index[idx]
                        P1 = pyche_df.loc[property_name, dinucleotide_1]
                        P2 = pyche_df.loc[property_name, dinucleotide_2]
                        sum_property_diff += (P1 - P2) ** 2
                    theta_j += sum_property_diff / len(pyche_list)
            theta_j /= (L - j - 1)
            theta.append(theta_j)

        # 计算PC-PseDNC-General特征向量
        feature_vector = []
        for k, dinucleotide in enumerate(pyche_df.columns):
            fk = dinucleotide_freq.get(dinucleotide, 0)
            denominator = sum(dinucleotide_freq.values()) + weight * sum(theta)
            if k < 16:
                # 1 ≤ k ≤ 16
                d_k = fk / denominator
            else:
                # 16 + 1 ≤ k ≤ 16 + λ
                d_k = weight * theta[k - 16 - 1] / denominator
            feature_vector.append(d_k)

        # 如果feature_vector中有Series对象，提取其数值
        feature_vector = [value.values[0] if isinstance(value, pd.Series) else value for value in feature_vector]

        all_feature_vectors.append(feature_vector)

    return np.array(all_feature_vectors)


def rna_pc_psednc_general(seq, request):
    lambda_value = int(request.form.get('pc_psednc_general_lambda_value'))
    weight = float(request.form.get('pc_psednc_general_weight'))
    dimensions = int(request.form.get('pc_psednc_general_Dimensions'))
    pcplist = request.form.getlist('pc_psednc_general_param3')
    pcplist = [int(x)-1 for x in pcplist]
    df = pd.read_excel("feature_extraction/pychemdata/Table 5.xlsx", sheet_name='Sheet1')
    data = pc_psednc_general(seq, df, pcplist, lambda_value, weight)

    if data.shape[1] > dimensions:
        if dimensions < data.shape[0]:
            pca = PCA(n_components=dimensions)
            data = pca.fit_transform(data)
    if data.shape[1] > dimensions:
        rp = GaussianRandomProjection(n_components=dimensions)
        data = rp.fit_transform(data)
    return data


if __name__ == '__main__':

    # 定义RNA序列
    rna_sequences = [
        "AUGGCUAUGG",  # 示例RNA序列1
        "CGAUACGUAA",  # 示例RNA序列2
    ]

    # 导入理化性质表格
    file_path = '../pychemdata/Table 5.xlsx'  # 替换为你的文件路径
    pyche_df = pd.read_excel(file_path, index_col=0)

    # 选择用于计算的理化性质索引列表（例如使用所有理化性质）
    pyche_list = [0, 1]  # 表示选取DataFrame中的Property1和Property2

    # 设置参数
    lambda_value = 3  # λ值
    weight = 0.5  # 权重因子，介于0到1之间

    # 调用函数
    feature_vectors = pc_psednc_general(rna_sequences, pyche_df, pyche_list, lambda_value, weight)

    # 打印特征向量
    for idx, vector in enumerate(feature_vectors):
        print(f"RNA序列{idx + 1}的特征向量: {len(vector)}")
