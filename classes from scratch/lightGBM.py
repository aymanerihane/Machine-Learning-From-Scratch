"""
Description: Implementation of LightGBM from scratch
"""
import numpy as np
from decision_tree import DecisionTree

class LightGBM :
    

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, bins=2):
        """
        Parameters
        ----------
        n_estimators : int, optional, default=100
            The number of boosting rounds.

        learning_rate : float, optional, default=0.1

        max_depth : int, optional, default=3
            Maximum depth of the individual trees.

        bins : int, optional, default=2
            Number of bins to discretize the continuous features.

        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.bins = bins
        self.trees = []

    def _mse_loss(self, y, y_pred):
        grad = 2*(y_pred - y)# gradient of the loss
        hess = 2*np.ones(y.shape[0]) # hessian is the derivative of the gradient
        return grad, hess
    
    def fit(self,X,y):
        #initialize predictions with the mean of the target values
        y_pred = np.full_like(y,np.mean(y, axis=0))

        for _ in range(self.n_estimators):
            #calculate the gradient and hessian of the loss
            grad,hess = self._mse_loss(y,y_pred)

            #fit a regression tree to the gradient
            tree = DecisionTree(min_samples_split=2,max_depth=self.max_depth,n_feats=None)
            tree.fit(X,grad,hess)
            self.trees.append(tree)

            #update the predictions with the learning rate
            y_pred = y_pred - self.learning_rate*tree.predict(X)

    def predict(self,X):
        #initialize predictions with zeros
        y_pred = np.zeros(X.shape[0],dtype=np.float64)

        #add predictions from all tree
        for tree in self.trees:
            y_pred += self.learning_rate*tree.predict(X)

        return y_pred
    

"""
# Example of usage
--------------------------------
# Dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Instantiate and train the model
lgbm = LightGBMFromScratch(n_estimators=10, learning_rate=0.1, max_depth=2, min_samples_split=2)
lgbm.fit(X, y)

# Predict
y_pred = lgbm.predict(X)
print("Predictions:", y_pred)

"""




    
