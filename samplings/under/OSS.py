from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter

def find_tomek_links(X, y):
    """
    Identify Tomek Links in a dataset.

    Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.

    Returns:
        mask (ndarray): A boolean mask indicating whether each sample is not in a Tomek Link.
    """
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    neighbors = knn.kneighbors(X, return_distance=False).flatten()

    mask = np.ones(len(X), dtype=bool)  # Start with all samples valid

    for i in range(len(X)):
        j = neighbors[i]
        if y[i] != y[j] and neighbors[j] == i:  # Mutual nearest neighbors with different labels
            # Remove the majority class sample in the Tomek Link
            if np.sum(y == y[i]) > np.sum(y == y[j]):  # i is majority class
                mask[i] = False
            else:  # j is majority class
                mask[j] = False

    return mask

def one_sided_selection(X, y, n_neighbors=1, max_iter=100):
    """
    One-Sided Selection (OSS) undersampling algorithm.

    Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.
        n_neighbors (int): Number of neighbors for CNN (default: 1).
        max_iter (int): Maximum iterations for CNN (default: 100).

    Returns:
        X_resampled (ndarray): Resampled feature matrix.
        y_resampled (ndarray): Resampled target vector.
    """
    # Step 1: Apply CNN
    unique_classes = np.unique(y)
    idx_initial = [np.where(y == cls)[0][0] for cls in unique_classes]
    X_resampled = X[idx_initial]
    y_resampled = y[idx_initial]

    idx_remaining = np.setdiff1d(np.arange(len(y)), idx_initial)
    X_remaining = X[idx_remaining]
    y_remaining = y[idx_remaining]

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    iteration = 0

    while len(X_remaining) > 0 and iteration < max_iter:
        iteration += 1
        knn.fit(X_resampled, y_resampled)

        misclassified_idx = []
        for i, (xi, yi) in enumerate(zip(X_remaining, y_remaining)):
            if knn.predict(xi.reshape(1, -1)) != yi:
                misclassified_idx.append(i)

        if len(misclassified_idx) == 0:
            break

        X_resampled = np.vstack((X_resampled, X_remaining[misclassified_idx]))
        y_resampled = np.hstack((y_resampled, y_remaining[misclassified_idx]))

        X_remaining = np.delete(X_remaining, misclassified_idx, axis=0)
        y_remaining = np.delete(y_remaining, misclassified_idx, axis=0)

    # Step 2: Apply Tomek Links
    mask = find_tomek_links(X_resampled, y_resampled)
    X_resampled = X_resampled[mask]
    y_resampled = y_resampled[mask]

    return X_resampled, y_resampled


def UnderOSS(request, X, y):
    label_counts = Counter(y.flatten())
    label_count_dict = dict(label_counts)
    print("Original counts:", label_count_dict)

    n_neighbors = int(request.form.get(f'OSS_n_neighbors'))
    x_resampled, y_resampled = one_sided_selection(X, y, n_neighbors=n_neighbors)

    unique, counts = np.unique(y_resampled, return_counts=True)
    sampled_distribution = dict(zip(unique, counts))

    print("采样后的类别分布:")
    for label, count in sampled_distribution.items():
        print(f"类别 {label}: {count} 个样本")

    return x_resampled, y_resampled


if __name__ == '__main__':
    # Create a multi-class imbalanced dataset
    X, y = make_classification(n_samples=2000, n_features=10, n_classes=4,
                               n_clusters_per_class=1, n_redundant=2,
                               weights=[0.6, 0.1, 0.1, 0.2],
                               n_informative=4, random_state=0)

    # Output original dataset class distribution
    print("Original dataset class distribution:", dict(zip(*np.unique(y, return_counts=True))))

    # Apply OSS for undersampling
    X_resampled, y_resampled = one_sided_selection(X, y, n_neighbors=1)

    # Output resampled dataset size and class distribution
    print("Resampled dataset size:", X_resampled.shape)
    print("Resampled dataset class distribution:", dict(zip(*np.unique(y_resampled, return_counts=True))))

