from collections import Counter
from scipy.interpolate import splrep, BSpline
from sklearn.preprocessing import MinMaxScaler
from numpy import polyfit, poly1d
import numpy as np


"""
1.	极值点偏差补偿聚类欠采样算法：该算法首先对需要进行欠采样的类别进行极值点偏差补偿聚类，而后根据样本到聚类中心的举例进行样本选择。（可以详细一点儿）
2.	都适用
3.	k：聚类数(大于0的整数) 
    p：拟合数据结点数(小于data.shape[1]-2, 至少为4，当data.shape[1]小于4的时候，无法使用该采样算法)
    Sampling Strategy{0：采样后的数目，1：采样后的数目}
"""


class FunctionalClustering:
    def __init__(self, data, k, p, smooth_factor=None):
        self.p = p
        self.data = data
        self.k = k
        self.smooth_factor = smooth_factor
        self.n = data.shape[0]
        self.dim = data.shape[1]
        self.normalized_data = None
        self.functions = []
        self.cluster_labels = None
        self.distances_to_centroid = None

    def normalize_data(self):
        """

        """
        scaler = MinMaxScaler()
        self.normalized_data = scaler.fit_transform(self.data)

    def fit_b_spline(self, x, y, k=3):
        num_knots = self.p
        t = np.linspace(x.min(), x.max(), num_knots)
        tck = splrep(x, y, k=k, t=t[1:-1], s=self.smooth_factor)
        # tck = splrep(x, y, k=k, t=t[1:-1])
        spline = BSpline(tck[0], tck[1], tck[2])
        return spline


    def fit_all_functions(self):
        """
        """
        pram = 1
        x = np.linspace(0, 1, self.dim)
        for i in range(self.n):
            if pram == 1:
                y = self.normalized_data[i]
                spl = self.fit_b_spline(x, y)
                self.functions.append(spl)

            if pram == 2:
                y = self.normalized_data[i]
                coeffs = polyfit(x, y, deg=4)
                poly_func = poly1d(coeffs)
                self.functions.append(poly_func)

    def find_extrema(self, f, num_points=10):
        x = np.linspace(0, 1, num_points)
        y = f(x)
        dy = np.gradient(y)
        extrema_indices = np.where(np.diff(np.sign(dy)))[0]

        if extrema_indices.size == 0:

            extrema = [x[0], x[-1]]
        else:

            extrema = x[extrema_indices]

        return extrema

    def calculate_similarity(self, f1, f2):

        x = np.linspace(0, 1, self.dim)
        y1 = f1(x)
        y2 = f2(x)

        numeric_distance = np.linalg.norm(y1 - y2)

        extrema1 = self.find_extrema(f1)
        extrema2 = self.find_extrema(f2)

        if len(extrema1) != len(extrema2):
            common_length = max(len(extrema1), len(extrema2))
            extrema1 = np.interp(np.linspace(0, 1, common_length), np.linspace(0, 1, len(extrema1)), extrema1)
            extrema2 = np.interp(np.linspace(0, 1, common_length), np.linspace(0, 1, len(extrema2)), extrema2)

        extrema_distance = np.linalg.norm(np.array(extrema1) - np.array(extrema2))
        total_distance = numeric_distance + extrema_distance

        return total_distance

    def calculate_Minkowski(self, f1, f2, p=2):
        x = np.linspace(0, 1, self.dim)
        y1 = f1(x)
        y2 = f2(x)
        distance = np.sum(np.abs(y1 - y2) ** p) ** (1 / p)
        return distance

    def calculate_Expansion_coefficient(self, f1, f2):
        coeffs1 = f1.c
        coeffs2 = f2.c
        return np.linalg.norm(coeffs1 - coeffs2)

    def kmeans_pp(self):

        distances = np.full(self.n, np.inf)
        centers = [np.random.randint(0, self.n)]

        for _ in range(1, self.k):
            for i in range(self.n):
                distances[i] = min(distances[i], self.calculate_similarity(self.functions[i], self.functions[centers[-1]]))
            centers.append(np.argmax(distances))

        labels = np.zeros(self.n)
        distances_to_centroid = np.zeros(self.n)
        for i in range(self.n):
            min_distance = np.inf
            for j in range(self.k):
                dist = self.calculate_similarity(self.functions[i], self.functions[centers[j]])
                if dist < min_distance:
                    min_distance = dist
                    labels[i] = j
            distances_to_centroid[i] = min_distance

        self.cluster_labels = labels
        self.distances_to_centroid = distances_to_centroid

    def run_clustering(self):

        self.normalize_data()
        self.fit_all_functions()
        self.kmeans_pp()
        results = np.vstack((np.arange(self.n), self.cluster_labels, self.distances_to_centroid)).T
        return results


def under_EPCCC(X, y, k, p,sampling_strategy):
    #采样后的数据和标签
    y = y.ravel()
    x_resampled, y_resampled = [], []
    if p > X.shape[1] - 2:
        p = X.shape[1] - 2
    # 遍历每一个类别和对应的采样数量
    for label, sample_count in sampling_strategy.items():
        # 获取当前类别的索引和数据
        indices = np.where(y == label)[0]
        X_label = X[indices]

        if sample_count <= len(indices):
            fc = FunctionalClustering(X_label, k, p)
            result = fc.run_clustering()
            indices = result[:, 2].argsort()
            sorted_data = X_label[indices]
            for i in range(sampling_strategy[label]):
                x_resampled.append(sorted_data[i])
                y_resampled.append(label)

    # 转换为NumPy数组
    x_resampled = np.array(x_resampled, dtype=float)
    y_resampled = np.array(y_resampled, dtype=float)

    return x_resampled, y_resampled



def UnderEPCCC(request, data, label):


    p = int(request.form.get(f'epdcc_p'))
    k = int(request.form.get(f'epdcc_k'))

    label_counts = Counter(label.flatten())
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    for l, n in label_counts.items():
        num = int(request.form.get(f'EPDCC{l}'))
        label_count_dict[l] = num
    print(label_count_dict)

    data, label = under_EPCCC(data, label, k, p, label_count_dict)

    return data, label




if __name__ == '__main__':
    # 生成每个类别的数据
    X = np.random.rand(100, 4)  # 类别 0 的样本
    y = np.array([0] * 10 + [1] * 20 + [2] * 70)
    k = 3
    p = 10
    sampling_strategy = {0: 10, 1: 15, 2: 30}

    x_resampled, y_resampled = under_EPCCC(X, y, k, p, sampling_strategy)
    print(x_resampled.shape)

    unique, counts = np.unique(y_resampled, return_counts=True)
    sampled_distribution = dict(zip(unique, counts))

    print("采样后的类别分布:")
    for label, count in sampled_distribution.items():
        print(f"类别 {label}: {count} 个样本")













