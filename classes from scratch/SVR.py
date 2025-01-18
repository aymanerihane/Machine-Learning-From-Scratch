import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SVR:
    def __init__(self, kernel="rbf", C=1.0, epsilon=0.1, degree=3, gamma=0.5, max_iter=1000, learning_rate=0.001):
        """
        parrameters:
        kernel: str, default="rbf"
            Kernel function to be used in the algorithm. Supported kernels are "linear", "poly", and "rbf".
        C: float, default=1.0
            Regularization parameter. The strength of the regularization is inversely proportional to C.
        epsilon: float, default=0.1
            Tolerance for the optimization problem.
        degree: int, default=3
            Degree of the polynomial kernel function ("poly").
        gamma: float, default=0.5
            Kernel coefficient for "rbf" kernel.
        max_iter: int, default=1000
            Maximum number of iterations for the optimization.
        learning_rate: float, default=0.001
            Learning rate for the gradient descent optimization.

        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.degree = degree
        self.gamma = gamma
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.alpha = None  # Lagrange multipliers

    # Kernel function
    def _kernel(self, X1, X2):
        """
        Parameters:
        X1: numpy.ndarray
            First input data matrix.
        X2: numpy.ndarray
            Second input data matrix.
        """
        if self.kernel == "linear": # Linear kernel
            return np.dot(X1, X2.T)
        elif self.kernel == "poly": # Polynomial kernel
            return (np.dot(X1, X2.T) + 1) ** self.degree
        elif self.kernel == "rbf": # Radial basis function kernel
            if X1.ndim == 1 and X2.ndim == 1: # if both are 1D arrays
                return np.exp(-self.gamma * np.linalg.norm(X1 - X2) ** 2)
            elif (X1.ndim > 1 and X2.ndim == 1) or (X1.ndim == 1 and X2.ndim > 1): # if one is 1D and the other is 2D array
                return np.exp(-self.gamma * np.linalg.norm(X1 - X2, axis=1) ** 2) 
            elif X1.ndim > 1 and X2.ndim > 1: # if both are 2D arrays
                return np.exp(-self.gamma * np.linalg.norm(X1[:, np.newaxis] - X2, axis=2) ** 2)
        else:
            raise ValueError("Unsupported kernel")

    # function to fit the model
    def fit(self, X, y):
        """
        Parameters:
        X: numpy.ndarray
            Input data matrix.
        y: numpy.ndarray
            Target values.
        """

        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)  # Lagrange multipliers for dual problem
        self.support_vectors = X 
        self.support_labels = y

        # Gradient descent-based optimization
        for _ in range(self.max_iter):
            for i in range(n_samples):
                prediction = self._predict_single(X[i])
                error = y[i] - prediction

                if abs(error) > self.epsilon:
                    self.alpha[i] += self.learning_rate * error

    # function to predict the output for a single input
    def _predict_single(self, x):
        kernel_values = self._kernel(self.support_vectors, x)
        return np.dot(self.alpha, kernel_values)

    # function to predict the output for multiple inputs
    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    # function to evaluate the model
    def evaluate(self, X, y, regression=False):
        y_pred = self.predict(X)
        if regression:
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            return rmse
        else:
            predictions = np.sign(y_pred)
            accuracy = np.mean(predictions == y)
            return accuracy
