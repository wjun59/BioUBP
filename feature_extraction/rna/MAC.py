import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


def rna_mac(seq, request):
    lag = int(request.form.get('mac_lag'))
    dimensions = int(request.form.get('mac_Dimensions'))
    pcplist = request.form.getlist('mac_param3')
    pcplist = [int(x)-1 for x in pcplist]
    df = pd.read_excel("feature_extraction/pychemdata/Table 6.xlsx", sheet_name='Sheet1')
    data = moran_autocorrelation(seq, df, pcplist, lag)

    if data.shape[1] > dimensions:
        if dimensions < data.shape[0]:
            pca = PCA(n_components=dimensions)
            data = pca.fit_transform(data)
    if data.shape[1] > dimensions:
        rp = GaussianRandomProjection(n_components=dimensions)
        data = rp.fit_transform(data)
    return data


def moran_autocorrelation(rna_sequences, pyche_df, pyche_list, lag):
    """
    计算多个RNA序列的Moran自相关特征

    参数：
    rna_sequences: list，RNA序列的数组
    pyche_df: pandas.DataFrame，包含理化性质的DataFrame（列为二核苷酸，行为理化性质）
    pyche_list: list，选择用于计算的理化性质的索引值列表
    lag: int，距离参数

    返回：
    list，每个RNA序列的MAC特征向量
    """
    all_mac_vectors = []

    # 遍历每个RNA序列
    for rna_sequence in rna_sequences:
        # 计算序列长度
        L = len(rna_sequence)

        # 确保序列长度足够大以计算给定的lag
        if L <= lag:
            raise ValueError("RNA序列长度必须大于lag值")

        # 初始化MAC值字典
        mac_values = {}

        # 遍历选定的理化性质索引
        for idx in pyche_list:
            # 获取理化性质名称
            property_name = pyche_df.index[idx]

            # 获取该理化性质的数值
            P_values = []
            for i in range(L - 1):
                dinucleotide = rna_sequence[i:i + 2]
                # 检查二核苷酸是否存在于DataFrame的列中
                if dinucleotide in pyche_df.columns:
                    P_values.append(pyche_df.loc[property_name, dinucleotide])
                else:
                    # 如果找不到该二核苷酸，使用默认值（例如0）
                    P_values.append(0)

            # 转换为numpy数组
            P_values = np.array(P_values)

            # 计算理化属性的均值
            P_mean = np.mean(P_values)

            # 计算分子部分
            numerator = 0
            for i in range(L - lag - 1):
                numerator += (P_values[i] - P_mean) * (P_values[i + lag] - P_mean)

            # 计算分母部分
            denominator = np.sum((P_values - P_mean) ** 2) / (L - 1)

            # 计算MAC值
            mac_value = (1 / (L - lag - 1)) * numerator / denominator if denominator != 0 else 0

            # 将结果存储在字典中
            mac_values[property_name] = mac_value

        # 将当前序列的MAC值字典转化为特征向量
        mac_vector = [mac_values[property_name] for property_name in mac_values]
        all_mac_vectors.append(mac_vector)

    return np.array(all_mac_vectors)


if __name__ == '__main__':
    # 导入理化性质表格
    file_path = '../pychemdata/Table 6.xlsx'  # 替换为你的文件路径
    pyche_df = pd.read_excel(file_path, index_col=0)

    # RNA序列数组和lag值
    rna_sequences = ["AUGCAGUCAUACG", "CUGCAGUCAUAG", "AUGCGAUCAUACG"]
    lag = 2

    # 选择用于计算的理化性质索引值列表
    pyche_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 根据你需要选择的理化性质的索引

    # 计算Moran自相关值
    mac_vectors = moran_autocorrelation(rna_sequences, pyche_df, pyche_list, lag)

    # 打印每个RNA序列的Moran自相关特征向量
    for i, vector in enumerate(mac_vectors):
        print(f"RNA序列 {i + 1} 的 Moran 自相关特征向量: {len(vector)}")
