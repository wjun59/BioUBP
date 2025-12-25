import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

def dna_onehot(seq, request):
    dimensions = int(request.form.get('onehot_Dimensions'))
    data = dna_onehot_encode_flatten(seq)

    if data.shape[1] > dimensions:
        if dimensions < data.shape[0]:
            pca = PCA(n_components=dimensions)
            data = pca.fit_transform(data)
    if data.shape[1] > dimensions:
        rp = GaussianRandomProjection(n_components=dimensions)
        data = rp.fit_transform(data)
    return data


def dna_onehot_encode_flatten(sequences):
    # 定义每个核苷酸的one-hot编码
    onehot_dict = {
        'A': np.array([1, 0, 0, 0]),
        'C': np.array([0, 1, 0, 0]),
        'G': np.array([0, 0, 1, 0]),
        'T': np.array([0, 0, 0, 1])
    }
    # 初始化一个空列表来存储每个序列的编码
    encoded_sequences = []

    # 遍历序列列表
    for seq in sequences:
        # 将序列转换为大写
        seq = seq.upper()
        # 使用列表推导式和onehot_dict字典来编码序列中的每个核苷酸
        # 并将所有编码向量连接成一个一维数组
        encoded_seq = np.concatenate([onehot_dict[nuc] for nuc in seq])
        # 将编码后的序列添加到列表中
        encoded_sequences.append(encoded_seq)

    # 将所有序列的编码合并成一个二维数组
    encoded_array = np.array(encoded_sequences)
    return encoded_array