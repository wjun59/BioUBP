import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter


def UnderERS(request, X, label):
    majority_class = int(request.form.get(f'ERS_majority_class'))
    # 希望减少哪个类别周围的样本
    y = label.ravel()
    X_resampled, y_resampled = ers_undersample(X, y, majority_class=majority_class)
    return X_resampled, y_resampled


def ers_undersample(X, y, majority_class):
    """
    Implements Edited Random Sampling (ERS) algorithm.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Label vector.
    majority_class : int
        The label of the majority class to be undersampled.

    Returns:
    X_resampled : array, shape (n_samples_new, n_features)
        The resampled feature matrix.
    y_resampled : array, shape (n_samples_new,)
        The resampled label vector.
    """
    # Find the majority class and the minority class
    majority_samples = X[y == majority_class]
    minority_samples = X[y != majority_class]

    # Find nearest neighbors of majority class samples
    nbrs = NearestNeighbors(n_neighbors=2).fit(minority_samples)
    distances, indices = nbrs.kneighbors(majority_samples)

    # Randomly select majority class samples to remove
    to_remove = []
    for i in range(len(majority_samples)):
        # Check if the majority class sample is close to a minority class sample
        if np.random.rand() < 0.5:  # Random threshold for deletion
            to_remove.append(i)

    # Create the resampled dataset by removing the selected samples
    X_resampled = np.delete(X, to_remove, axis=0)
    y_resampled = np.delete(y, to_remove, axis=0)

    return X_resampled, y_resampled


if __name__ == '__main__':
    # Example usage:
    from sklearn.datasets import make_classification

    # Generate a toy dataset with imbalanced classes
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_clusters_per_class=2, n_informative=3,
                               weights=[0.7, 0.2, 0.1], random_state=42)

    # Apply Edited Random Sampling (ERS) to undersample the majority class (label 0)
    X_resampled, y_resampled = ers_undersample(X, y, majority_class=0)

    # Check the class distribution after resampling
    print("Original class distribution:", Counter(y))
    print("Resampled class distribution:", Counter(y_resampled))

