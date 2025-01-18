"""
Initialization (__init__ method):

n_components: Number of principal components to keep.
components: Principal components (eigenvectors).
mean: Mean of the dataset.
explained_variance_ratio_: Ratio of variance explained by each principal component.
Fitting the model (fit method):

Mean centering the data.
Computing the covariance matrix.
Calculating eigenvalues and eigenvectors of the covariance matrix.
Sorting eigenvectors by eigenvalues in descending order.
Calculating the explained variance ratio.
Storing the top n_components eigenvectors.
Transforming the data (transform method):

Projecting the data onto the principal components.
Utility functions:

load_data: Loads data from a CSV file.
plot_cumulative_variance: Plots the cumulative explained variance.
find_optimal_components: Finds the optimal number of components to explain a given variance threshold.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Covariance matrix (function needs samples as columns)
        cov = np.cov(X_centered, rowvar=False)

        # Eigenvalues, eigenvectors
        eigenvalues, eigenvectors = eigh(cov) # eigh is used for symmetric matrices

        # Sort eigenvectors by eigenvalues in descending order
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[:, idxs]

        # Explained variance ratio
        self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)

        # Store the first n eigenvectors (components)
        if self.n_components is not None:
            self.components = eigenvectors[:, :self.n_components]
        else:
            cumulative_variance = np.cumsum(self.explained_variance_ratio_)
            self.n_components = find_optimal_components(cumulative_variance)
            self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Project data onto the principal components
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)


def load_data(file_path):
    """Load the data from a CSV file."""
    data = pd.read_csv(file_path)
    # Assuming the target variable is the last column and the rest are features
    X = data.iloc[:, :-1].values  # Features (all columns except last)
    y = data.iloc[:, -1].values   # Target (last column)
    return X, y


def plot_cumulative_variance(explained_variance_ratio_):
    """Plot the cumulative explained variance against the number of components."""
    cumulative_variance = np.cumsum(explained_variance_ratio_)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance by Number of Principal Components")
    plt.grid()
    plt.show()


def find_optimal_components(cumulative_variance, threshold=0.9998):
    """Find the optimal number of components to explain a certain variance threshold."""
    optimal_components = np.argmax(cumulative_variance >= threshold) + 1  # Adding 1 for index correction
    return optimal_components
