from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def UnderCNN(request, X, label):
    k = int(request.form.get("CNN_k"))  # Number of neighbors to consider (default: 1).
    max_iter = int(request.form.get("CNN_max_iter"))

    y = label.ravel()
    resampled_data, resampled_labels = condensed_nearest_neighbour(X, y, n_neighbors=k, max_iter=max_iter)
    return resampled_data, resampled_labels


def condensed_nearest_neighbour(X, y, n_neighbors=1, max_iter=100):
    """
    Condensed Nearest Neighbour (CNN) undersampling algorithm for multi-class datasets.

    Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.
        n_neighbors (int): Number of neighbors to consider (default: 1).
        max_iter (int): Maximum number of iterations (default: 100).

    Returns:
        X_resampled (ndarray): Resampled feature matrix.
        y_resampled (ndarray): Resampled target vector.
    """
    # Initialize the condensed set S with all unique classes (at least one sample per class)
    unique_classes = np.unique(y)
    idx_initial = [np.where(y == cls)[0][0] for cls in unique_classes]
    X_resampled = X[idx_initial]
    y_resampled = y[idx_initial]

    # Remaining samples
    idx_remaining = np.setdiff1d(np.arange(len(y)), idx_initial)
    X_remaining = X[idx_remaining]
    y_remaining = y[idx_remaining]

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    iteration = 0

    while len(X_remaining) > 0 and iteration < max_iter:
        iteration += 1
        # Fit KNN on the current condensed set
        knn.fit(X_resampled, y_resampled)

        # Predict remaining samples
        misclassified_idx = []
        for i, (xi, yi) in enumerate(zip(X_remaining, y_remaining)):
            if knn.predict(xi.reshape(1, -1)) != yi:
                misclassified_idx.append(i)

        if len(misclassified_idx) == 0:
            break  # Exit if no more misclassified samples

        # Add misclassified samples to the condensed set
        X_resampled = np.vstack((X_resampled, X_remaining[misclassified_idx]))
        y_resampled = np.hstack((y_resampled, y_remaining[misclassified_idx]))

        # Remove these samples from the remaining set
        X_remaining = np.delete(X_remaining, misclassified_idx, axis=0)
        y_remaining = np.delete(y_remaining, misclassified_idx, axis=0)

    return X_resampled, y_resampled


if __name__ == '__main__':
    # Create a multi-class imbalanced dataset
    X, y = make_classification(n_samples=2000, n_features=10, n_classes=2,
                               n_clusters_per_class=1, n_redundant=2,
                               weights=[0.6, 0.4],
                               n_informative=4, random_state=0)

    # Output original dataset class distribution
    print("Original dataset class distribution:", dict(zip(*np.unique(y, return_counts=True))))

    # Apply CNN for undersampling
    X_resampled, y_resampled = condensed_nearest_neighbour(X, y, n_neighbors=2)

    # Output resampled dataset size and class distribution
    print("Resampled dataset size:", X_resampled.shape)
    print("Resampled dataset class distribution:", dict(zip(*np.unique(y_resampled, return_counts=True))))

