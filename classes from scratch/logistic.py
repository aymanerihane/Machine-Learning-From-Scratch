import numpy as np
from sklearn.model_selection import train_test_split

class Logistic():
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_list = []
        self.losses = []

    # Sigmoid function
    def _sigmoid(self, z):
        """
            Formula: 1 / (1 + e^(-z))
        """
        # Clip values of z to avoid overflow in exp function
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))


    # Prediction function
    def _predict(self, X, weights):
        z = np.dot(X, weights)
        return self._sigmoid(z)

    # Loss function (Binary Cross-Entropy)
    def _compute_loss(self, y, y_pred):
        """
            Formula: -1/N * Σ(y * log(y_pred) + (1 - y) * log(1 - y_pred))
        """
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Avoid log(0)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def fit(self, X, y):
        return self.one_vs_all(X, y)

    # Gradient descent step
    def _gradient_descent(self, X, y, weights):
        N = len(y)
        y_pred = self._predict(X, weights)
        gradient = np.dot(X.T, (y_pred - y)) / N
        weights -= self.learning_rate * gradient
        return weights

    # Train logistic regression for binary classification
    def train_logistic_regression(self, X, y_binary):
        weights = np.zeros(X.shape[1])  # Initialize weights
        losses = []

        # Gradient descent loop
        for epoch in range(self.epochs):
            y_pred = self._predict(X, weights)
            loss = self._compute_loss(y_binary, y_pred)
            weights = self._gradient_descent(X, y_binary, weights)
            self.weights_list.append(weights)  # Store weights for each epoch

            # Track the loss every 100 epochs
            if epoch % 100 == 0:
                losses.append(loss)
                print(f"Epoch {epoch}: Loss = {loss}")

        return weights, losses

    # One-vs-All Logistic Regression
    def one_vs_all(self, X, y):
        num_classes = len(np.unique(y))  # Number of classes
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Train a logistic regression model for each class (One-vs-All)
        for i in range(num_classes):
            y_binary = np.where(y_train == i, 1, 0)  # Create binary target for class i
            weights, losses = self.train_logistic_regression(X_train, y_binary)
            self.weights_list.append(weights)  # Store weights for each class
            self.losses.append(losses)  # Track losses

        return self.weights_list, self.losses

    # Split data into training and testing sets
    def split_data(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)

    # Function to calculate the error rate
    @staticmethod
    def error_rate(y_true, y_pred):
        incorrect = np.sum(y_true != y_pred)  # Count incorrect predictions
        total = len(y_true)  # Total predictions
        return incorrect / total  # Error rate

    # Predict class for new samples (multiclass)
    def predict_multiclass(self, X):
        if not self.weights_list:
            raise ValueError("Model has not been trained yet. Call 'one_vs_all' first.")
        
        # Calculate probabilities for each class
        probabilities = np.array([self._predict(X, weights) for weights in self.weights_list]).T
        return np.argmax(probabilities, axis=1)  # Return class with highest probability

    def score(self,y,y_pred):
        return np.mean(y == y_pred)