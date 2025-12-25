import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection


def dna_tcc(seq, request):
    lag = int(request.form.get('tcc_lag'))
    dimensions = int(request.form.get('tcc_Dimensions'))
    pcplist = request.form.getlist('tcc_param3')
    pcplist = [int(x) for x in pcplist]
    df = pd.read_excel("feature_extraction/pychemdata/Table 2.xlsx", sheet_name='Sheet1')
    data = tcc_feature_extraction(seq, lag, pcplist, df)

    if data.shape[1] > dimensions:
        if dimensions < data.shape[0]:
            pca = PCA(n_components=dimensions)
            data = pca.fit_transform(data)
    if data.shape[1] > dimensions:
        rp = GaussianRandomProjection(n_components=dimensions)
        data = rp.fit_transform(data)
    return data


def tcc_feature_extraction(sequences, lag, pyche_list, pyche_df):
    all_features = []

    # 提取选定的物理化学指标的行
    selected_indices = pyche_df.iloc[[i - 1 for i in pyche_list]]
    n_indices = len(selected_indices)  # 选定的指标数量

    # 计算每个选定物理化学指标的平均值
    avg_values = {}
    for idx, row in selected_indices.iterrows():
        avg_values[idx] = sum([row[sequences[0][i] + sequences[0][i + 1] + sequences[0][i + 2]]
                               for i in range(len(sequences[0]) - 2)]) / (len(sequences[0]) - 2)

    for seq in sequences:
        L = len(seq)
        # 计算TCC特征
        tcc_features = []

        for i in range(n_indices):
            for j in range(n_indices):
                if i != j:  # 确保使用不同的物理化学属性
                    row1 = selected_indices.iloc[i]
                    row2 = selected_indices.iloc[j]
                    avg_u1 = avg_values[selected_indices.index[i]]
                    avg_u2 = avg_values[selected_indices.index[j]]

                    for l in range(1, lag + 1):
                        tcc_sum = 0
                        for k in range(L - l - 2):
                            p1 = row1[seq[k] + seq[k + 1] + seq[k + 2]]
                            p2 = row2[seq[k + l] + seq[k + l + 1] + seq[k + l + 2]]
                            tcc_sum += (p1 - avg_u1) * (p2 - avg_u2)
                        tcc = tcc_sum / (L - l - 2)
                        tcc_features.append(tcc)

        all_features.append(tcc_features)

    return np.array(all_features)

