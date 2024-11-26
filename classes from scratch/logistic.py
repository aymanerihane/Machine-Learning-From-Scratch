import numpy as np
from sklearn.model_selection import train_test_split

class LogisticRegression:

    def __init__(self, X, y, learning_rate=0.01, epochs=1000, num_labels=10):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_labels = num_labels
        self.weights_list = []
        self.losses = []

    # Step 1: Sigmoid Function
    def _sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    # Step 2: Prediction Function
    def _predict(self,X, weights):
        z = np.dot(X, weights)
        return self._sigmoid(z)

    # Step 3: Loss Function (Binary Cross-Entropy)
    def _compute_loss(y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # Step 4: Gradient Descent
    def _gradient_descent(self,X, y, weights):
        N = len(y)
        y_pred = self._predict(X, weights)
        gradient = np.dot(X.T, (y_pred - y)) / N
        weights -= self.learning_rate * gradient
        return weights


    # Step 5: Training Function
    def _train_logistic_regression(self):
        # Initialize weights
        weights = np.zeros(self.X.shape[1])
        los = []
        # Gradient Descent
        for epoch in range(self.epochs):
            y_pred = self._predict(self.X, weights)
            losss = self._compute_loss(self.y, y_pred)
            weights = self._gradient_descent(self.X, self.y, weights, self.learning_rate)

            # Print the loss every 100 epochs for tracking
            if epoch % 100 == 0:
                los.append(losss)
                print(f'aEpoch {epoch}: Loss = {losss}')

        return weights,los
    
    #ONE VS ALL
    def one_vs_all(self):
        # One-vs-Rest Strategy for multiclass classification
        num_classes = len(np.unique(self.y))
        
        
        X_train, X_test, y_train, y_test = self.split_data(self.X,self.y)

        # Train a logistic regression model for each class
        for i in range(num_classes):
            # Create binary target variable: 1 if current class, 0 otherwise
            y_binary = np.where(y_train == i, 1, 0)
            weights, los = self._train_logistic_regression(X_train, y_binary, self.learning_rate, self.epochs)
            self.weights_list.append(weights)
            self.losses.append(los)
        
        return self.weights_list, self.losses

    def split_data(self,X,y):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    # Function to calculate error rate
    def error_rate(y_true, y_pred):
        incorrect = np.sum(y_true != y_pred)  # Count incorrect predictions
        print(incorrect)
        total = len(y_true)
        print(total)                   # Total predictions
        return incorrect / total               # Calculate error rate
    
    # Function to predict class for new samples
    def predict_multiclass(self,X):
        if not self.weights_list:
            self.one_vs_all()

        probabilities = np.array([self._predict(X, weights) for weights in self.weights_list]).T
        return np.argmax(probabilities, axis=1)

