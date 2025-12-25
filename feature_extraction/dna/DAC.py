import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import GaussianRandomProjection


def dna_dac(seq, request):

    lag = int(request.form.get('dac_lag'))
    dimensions = int(request.form.get('dac_Dimensions'))
    pcplist = request.form.getlist('dac_param3')
    pcplist = [int(x) for x in pcplist]
    df = pd.read_excel("feature_extraction/pychemdata/Table 1.xlsx", sheet_name='Sheet1')
    data = dac_feature_extraction(seq, lag, pcplist, df)

    if data.shape[1] > dimensions:
        if dimensions < data.shape[0]:
            pca = PCA(n_components=dimensions)
            data = pca.fit_transform(data)

    if data.shape[1] > dimensions:
        rp = GaussianRandomProjection(n_components=dimensions)
        data = rp.fit_transform(data)

    return data


def dac_feature_extraction(sequences, lag, pyche_list, pyche_df):
    all_features = []
    for seq in sequences:
        L = len(seq)
        # Extract selected physicochemical indices
        selected_indices = pyche_df.iloc[[i - 1 for i in pyche_list]]
        # Calculate average value for each selected physicochemical index
        avg_values = {}
        for idx, row in selected_indices.iterrows():
            avg_values[idx] = sum([row[seq[i] + seq[i + 1]] for i in range(L - 1)]) / (L - 1)
        # Calculate DAC features for each physicochemical index and lag
        dac_features = []
        for idx, row in selected_indices.iterrows():
            avg_u = avg_values[idx]
            for l in range(1, lag + 1):
                dac_sum = 0
                for i in range(L - l - 1):
                    p1 = row[seq[i] + seq[i + 1]]
                    p2 = row[seq[i + l] + seq[i + l + 1]]
                    dac_sum += (p1 - avg_u) * (p2 - avg_u)
                dac = dac_sum / (L - l - 1)
                dac_features.append(dac)
        all_features.append(dac_features)
    return np.array(all_features)


if __name__ == '__main__':
    sequences = ["ATTTGCACCGATTTGCACCG", "ATTTGCACCGATTTGCACCG", "ATTGCACCGATCTGCACCG" ]
    sequences2 = ["ATTT", "ATTT", "ACCG"]
    lag = 3
    df = pd.read_excel("feature_extraction/pychemdata/Table 1.xlsx", sheet_name='Sheet1')
    pcplist = [1, 2, 3, 10, 22]  # 这个是Table1中选择的理化性质(代表第一个，第二个。。。 从1开始)

    data = dac_feature_extraction(sequences, lag, pcplist, df)

