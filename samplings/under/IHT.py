from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
# 只是适用于二分类


# Update IHT function to support model selection
def instance_hardness_threshold(X, y, model_type=1, threshold=0.5):
    """
    Instance Hardness Threshold (IHT) undersampling algorithm with model selection.

    Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.
        model_type (int): Type of model to use for calculating instance hardness.
                          1 = Logistic Regression
                          2 = SVM (probability=True)
                          3 = Random Forest (default)
                          4 = K-Nearest Neighbors
        threshold (float): Instance hardness threshold for majority class retention.

    Returns:
        X_resampled (ndarray): Resampled feature matrix.
        y_resampled (ndarray): Resampled target vector.
    """
    # Select model based on input
    if model_type == 1:
        model = LogisticRegression(random_state=0, max_iter=1000)
    elif model_type == 2:
        model = SVC(probability=True, random_state=0)
    elif model_type == 3:
        model = RandomForestClassifier(random_state=0, n_estimators=100)
    elif model_type == 4:
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("Invalid model type. Choose from 1 (Logistic Regression), 2 (SVM), 3 (Random Forest), or 4 (KNN).")

    # Split into train and test to avoid data leakage in evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Train the model
    model.fit(X_train, y_train)

    # Predict probabilities on the entire dataset
    y_proba = model.predict_proba(X)

    # Calculate instance hardness (1 - P(y | X))
    hardness = 1 - y_proba[np.arange(len(y)), y]

    # Separate majority and minority classes
    unique_classes, class_counts = np.unique(y, return_counts=True)
    majority_class = unique_classes[np.argmax(class_counts)]

    # Retain samples based on the hardness threshold
    mask = (y != majority_class) | (hardness > threshold)
    X_resampled = X[mask]
    y_resampled = y[mask]

    return X_resampled, y_resampled


def UnderIHT(request, X, label):

    model_type = int(request.form.get("IHT_model_type"))
    threshold = float(request.form.get("IHT_threshold"))
    y = label.ravel()
    resampled_data, resampled_labels = instance_hardness_threshold(X, y, model_type=model_type, threshold=threshold)
    return resampled_data, resampled_labels


if __name__ == '__main__':
    # Create an example dataset with imbalanced target classes
    X, y = make_classification(n_samples=2000, n_features=10, n_classes=2,
                               n_clusters_per_class=1, n_redundant=2,
                               weights=[0.8, 0.2], n_informative=3, random_state=0)

    # Output original dataset class distribution
    print("Original dataset class distribution:", dict(zip(*np.unique(y, return_counts=True))))

    # Model type (1 = Logistic Regression, 2 = SVM, 3 = Random Forest, 4 = KNN)
    model_type = 1  # Change this to test different models

    # Apply IHT for undersampling
    X_resampled, y_resampled = instance_hardness_threshold(X, y, model_type=model_type, threshold=0.01)

    # Output resampled dataset size and class distribution
    print("Resampled dataset size:", X_resampled.shape)
    print("Resampled dataset class distribution:", dict(zip(*np.unique(y_resampled, return_counts=True))))
