"""
Description: This file contains the class EFSA that will be used to apply the EFSA algorithm to the dataset.
it will contain the following methods:
- correlation_filter: this method will apply the correlation filter to the dataset.
- LasoCv: this method will apply the LassoCV method to the dataset.
-....
"""


import numpy as np

class EFSA:
    def __init__(self, correlation_threshold=0.3, lasso_penalty=0.01, n_features=None):
        """
        Parameters:
        - correlation_threshold: Threshold for feature correlation with the target variable.
        - lasso_penalty: Regularization parameter for Lasso-like feature selection.
        - n_features: Number of features to select in the ensemble step.
        """
        self.correlation_threshold = correlation_threshold
        self.lasso_penalty = lasso_penalty
        self.n_features = n_features
        self.selected_features_ = None

    def _correlation_selection(self, X, y):
        """
        Select features based on correlation with the target.
        """
        correlations = []
        for i in range(X.shape[1]):
            feature = X[:, i]
            corr = np.corrcoef(feature, y)[0, 1]  # Pearson correlation
            correlations.append((i, abs(corr)))

        # Select features above the correlation threshold
        selected = [i for i, corr in correlations if corr >= self.correlation_threshold]
        return selected, correlations

    def _lasso_selection(self, X, y):
        """
        Simplified Lasso-like feature selection using gradient descent.
        """
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)  # Initialize weights to zero
        lr = 0.01  # Learning rate
        n_iterations = 1000

        for _ in range(n_iterations):
            y_pred = np.dot(X, weights)
            residual = y - y_pred

            # Compute gradients for weights
            gradients = -2 * np.dot(X.T, residual) / n_samples + self.lasso_penalty * np.sign(weights)

            # Update weights
            weights -= lr * gradients

        # Select features with non-zero weights
        selected = [i for i in range(n_features) if abs(weights[i]) > 1e-4]
        return selected, weights

    def _mutual_info_selection(self, X, y):
        """
        Approximation of mutual information using entropy-based computation.
        """
        def entropy(arr):
            probs = np.bincount(arr) / len(arr)
            return -np.sum(p * np.log2(p) for p in probs if p > 0)

        mutual_infos = []
        for i in range(X.shape[1]):
            feature = X[:, i]
            joint_entropy = entropy(np.c_[feature, y].view(np.int32).flatten())
            mi = entropy(feature) + entropy(y) - joint_entropy
            mutual_infos.append((i, mi))

        # Sort by mutual information
        sorted_features = sorted(mutual_infos, key=lambda x: x[1], reverse=True)
        return [i for i, _ in sorted_features], mutual_infos

    def fit(self, X, y):
        """
        Runs the ensemble feature selection process.
        """
        # Correlation-based selection
        corr_selected, corr_scores = self._correlation_selection(X, y)
        print(f"Correlation-selected features: {corr_selected}")

        # Lasso-based selection
        lasso_selected, lasso_weights = self._lasso_selection(X, y)
        print(f"Lasso-selected features: {lasso_selected}")

        # Mutual information-based selection
        mi_selected, mi_scores = self._mutual_info_selection(X, y)
        print(f"Mutual information-selected features: {mi_selected[:5]} (top 5)")

        # Combine selected features and score them
        combined_scores = {}
        for feature in set(corr_selected + lasso_selected + mi_selected):
            combined_scores[feature] = (
                next((score for i, score in corr_scores if i == feature), 0) +
                (lasso_weights[feature] if feature in lasso_selected else 0) +
                next((score for i, score in mi_scores if i == feature), 0)
            )

        # Sort features by combined score and select top n_features
        sorted_features = sorted(combined_scores, key=combined_scores.get, reverse=True)
        self.selected_features_ = sorted_features[:self.n_features] if self.n_features else sorted_features

    def transform(self, X):
        """
        Transforms the dataset to include only the selected features.
        """
        assert self.selected_features_ is not None, "The EFSA model has not been fitted yet."
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        """
        Fits the model and transforms the dataset in a single step.
        """
        self.fit(X, y)
        return self.transform(X)
