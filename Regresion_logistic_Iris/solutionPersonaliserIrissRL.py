import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap

# Load the Iris dataset
dataset = sns.load_dataset('iris')

# Prepare the features and target variable
X = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = dataset['species'].values

# Convert categorical labels to numeric values (One-vs-Rest)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert species names to integers

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Add bias term to training data
X_train_with_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test_with_bias = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Prediction Function
def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z)

# Loss Function (Binary Cross-Entropy)
def compute_loss(y, y_pred):
    return -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))

# Gradient Descent
def gradient_descent(X, y, weights, learning_rate):
    N = len(y)
    y_pred = predict(X, weights)
    gradient = np.dot(X.T, (y_pred - y)) / N
    weights -= learning_rate * gradient
    return weights

# One-vs-Rest Logistic Regression Training
def train_logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    # OvR - Train one model per class
    n_classes = len(np.unique(y))
    weights = np.zeros((n_classes, X.shape[1]))

    for c in range(n_classes):
        y_binary = (y == c).astype(int)  # Create a binary target for class `c`
        w = np.zeros(X.shape[1])  # Initialize weights for class `c`

        # Gradient Descent for each class
        for epoch in range(epochs):
            y_pred = predict(X, w)
            loss = compute_loss(y_binary, y_pred)
            w = gradient_descent(X, y_binary, w, learning_rate)

            if epoch % 100 == 0:
                print(f'Class {c} - Epoch {epoch}: Loss = {loss}')

        weights[c] = w  # Store the weights for class `c`

    return weights

# Train the model using OvR
weights = train_logistic_regression(X_train_with_bias, y_train, learning_rate=0.1, epochs=1000)
print("Trained weights for each class:\n", weights)

# Decision Boundary Visualization (for first two features only)
cm = plt.cm.Blues
cm_bright = ListedColormap(['#FFFF00', '#00FFFF', '#FF00FF'])

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Training and Testing Data Points
axes[0].set_title("Training and Testing Data Points")
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k', label="Training")
axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k', label="Testing")
axes[0].legend()
axes[0].set_xticks(()); axes[0].set_yticks(())

# Decision Boundary with Probability Heatmap
h = .02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_plot = np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel(), np.zeros((xx.ravel().shape[0], 2))]

# Predict for each class and get the decision boundary
Z = np.array([predict(X_plot, weights[c]) for c in range(weights.shape[0])])
Z = np.argmax(Z, axis=0).reshape(xx.shape)

# Decision Boundary with Probability Heatmap
axes[1].contourf(xx, yy, Z, cmap=cm, alpha=0.65)
axes[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.1)
axes[1].set_title("Decision Boundary with Probability Heatmap")
axes[1].set_xticks(()); axes[1].set_yticks(())

# Rounded Decision Boundary
axes[2].set_title("Rounded Decision Boundary")
axes[2].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
axes[2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.1)
axes[2].contourf(xx, yy, Z, cmap=cm, alpha=0.65)
axes[2].set_xticks(()); axes[2].set_yticks(())

plt.tight_layout()
plt.show()
