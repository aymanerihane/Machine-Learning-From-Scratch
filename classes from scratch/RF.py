"""
Description: Implementation of Random Forest from scratch
"""

from decision_tree import DecisionTree
import numpy as np
from collections import Counter
import pandas as pd

# Classe de forêt aléatoire
def bootstrap_sample(X,y):
    X = np.array(X)
    y = np.array(y)
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples,n_samples,replace=True)
    return X[idxs],y[idxs]

def most_common_label(y):
    y = list(y)
    counter = Counter(y) # compter les occurences de chaque label
    most_common = counter.most_common(1)[0][0]
    return most_common

class RandomForest:
    def __init__(self,n_trees=100,min_samples_split=2,max_depth=100,n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self,X,y):
        X = pd.DataFrame(X)
        y = pd.Series(y)
        self.trees = []
        for i in range(self.n_trees):
            # if (i+1)%10 == 0:
            print(f"Fitting tree {i+1}...")
            tree = DecisionTree(max_depth=self.max_depth)
            X_sample,y_sample = bootstrap_sample(X,y)
            X_sample = pd.DataFrame(X_sample)
            y_sample = pd.Series(y_sample)
            tree.fit(X_sample,y_sample)
            self.trees.append(tree)

    def score(self,y,y_pred):
        return np.mean(y == y_pred)

    def predict(self,X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds,0,1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)