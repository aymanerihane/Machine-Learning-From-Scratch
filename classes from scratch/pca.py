"""
Description: This file contains the PCA class which is used to reduce the dimensionality of the data.

"""

import numpy as np


# variance
class PCA:
    def __init__(self,n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self,X):
        #mean
        self.mean = np.mean(X, axis=0)
        X = X -self.mean

        #covariance
        cov = np.cov(X.T)

        #eigenvectors, eigenvalues
        eiganvalues , eiganvectors = np.linalg.eig(cov)

        #sort eigenvectors
        eiganvectors = eiganvectors.T
        idxs = np.argsort(eiganvalues)[:,-1]
        eiganvalues = eiganvalues[idxs]
        eiganvectors = eiganvectors[idxs]

        #store first n eigvectors
        self.components = eiganvectors[0:self.n_components]

    def transform(self,X):
        #project data
        X= X -self.mean
        return np.dot(X,self.components)
