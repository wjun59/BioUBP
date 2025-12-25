import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection

def dna_tac(seq, request):
    lag = int(request.form.get('tac_lag'))
    dimensions = int(request.form.get('tac_Dimensions'))
    pcplist = request.form.getlist('tac_param3')
    pcplist = [int(x) for x in pcplist]
    df = pd.read_excel("feature_extraction/pychemdata/Table 2.xlsx", sheet_name='Sheet1')
    data = tac_feature_extraction(seq, lag, pcplist, df)

    if data.shape[1] > dimensions:
        if dimensions < data.shape[0]:
            pca = PCA(n_components=dimensions)
            data = pca.fit_transform(data)
    if data.shape[1] > dimensions:
        rp = GaussianRandomProjection(n_components=dimensions)
        data = rp.fit_transform(data)
    return data


def tac_feature_extraction(sequences, lag, pyche_list, pyche_df):
    all_features = []
    for seq in sequences:
        L = len(seq)
        # 提取所选物理化学指数
        selected_indices = pyche_df.iloc[[i - 1 for i in pyche_list]]
        # 计算每个选定物理化学指标的平均值
        avg_values = {}
        for idx, row in selected_indices.iterrows():
            avg_values[idx] = sum([row[seq[i] + seq[i + 1] + seq[i + 2]] for i in range(L - 2)]) / (L - 2)

        # 计算 TAC 特征值
        tac_features = []
        for idx, row in selected_indices.iterrows():
            avg_u = avg_values[idx]
            for l in range(1, lag + 1):
                tac_sum = 0
                for i in range(L - l - 2):
                    p1 = row[seq[i] + seq[i + 1] + seq[i + 2]]
                    p2 = row[seq[i + l] + seq[i + l + 1] + seq[i + l + 2]]
                    tac_sum += (p1 - avg_u) * (p2 - avg_u)
                tac = tac_sum / (L - l - 2)
                tac_features.append(tac)
        all_features.append(tac_features)

    return np.array(all_features)



