import numpy as np
from leafWiseGrowth import LeafWiseTree
# Define the LightFBM class (LightGBM-like)
class LightFBM:
    def __init__(self, n_estimators=3, learning_rate=0.1, max_depth=3, n_jobs=4):
        self.trees = []
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.initial_prediction = 0.5  # Starting prediction for logistic regression
        self.n_jobs = n_jobs  # Number of parallel threads to use

    def fit(self, X, y):
        # Initialize the prediction with constant value
        predictions = np.full_like(y, self.initial_prediction, dtype=float)

        for i in range(self.n_estimators):
            # Compute the gradients and Hessians based on current predictions
            gradients, hessians = self._compute_grad_hess(y, predictions)

            tree = LeafWiseTree(num_bins=60)  # Initialize the custom tree
            print(f"Fitting tree {i + 1}")
            tree.grow_tree(X, gradients, hessians, max_depth=self.max_depth, n_jobs=self.n_jobs)  # Grow the custom tree with parallelization
            self.trees.append(tree)

            # Update predictions (Gradient Boosting update step)
            tree_predictions = self._predict_tree(tree.root, X)
            predictions += self.learning_rate * tree_predictions  # Update with learning rate

    def _compute_grad_hess(self, y, predictions):
        # Compute gradients and Hessians for logistic regression (example)
        errors = y - predictions
        gradients = errors  # Gradient is the error
        hessians = np.ones_like(errors)  # Hessian is constant (1) for simplicity
        return gradients, hessians

    def _predict_tree(self, node, X):
        if node.is_leaf:
            return np.full(X.shape[0], node.prediction)  # Return the leaf prediction
        else:
            left_indices = X[:, node.value] < node.threshold # node.value is the feature index
            right_indices = X[:, node.value] >= node.threshold
            predictions = np.zeros(X.shape[0])
            predictions[left_indices] = self._predict_tree(node.left, X[left_indices])
            predictions[right_indices] = self._predict_tree(node.right, X[right_indices])
            return predictions

    def predict(self, X):
        # Final prediction is the sum of all trees' predictions, scaled by learning rate
        predictions = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            tree_predictions = self._predict_tree(tree.root, X)
            predictions += self.learning_rate * tree_predictions # Update with learning rate (shrinkage)
        return (predictions > 0.5).astype(int)

    
    def score(self,y,y_pred):
        return np.mean(y == y_pred)