import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Define the TreeNode class
class TreeNode:
    def __init__(self, value=None):
        self.value = value  # The feature index for the split
        self.threshold = None  # The threshold value for the split
        self.left = None  # Left child
        self.right = None  # Right child
        self.is_leaf = True  # Indicates if the node is a leaf
        self.prediction = None  # Prediction value for leaf nodes

# Define the LeafWiseTree class (Histogram-Based LightGBM Tree)
class LeafWiseTree:
    def __init__(self, num_bins=256):
        self.root = TreeNode()  # Initialize the root of the tree
        self.num_bins = num_bins  # Number of bins for histogram-based splitting

    def grow_tree(self, X, gradients, hessians, max_depth=3, n_jobs=4):
        # Pre-bin the features for histogram-based splitting
        self.binned_X, self.bin_edges = self._bin_features(X)
        self._grow_node(self.root, self.binned_X, gradients, hessians, depth=0, max_depth=max_depth, n_jobs=n_jobs)

    def _bin_features(self, X):
        # Bin each feature into discrete bins
        bin_edges = []
        binned_X = np.zeros_like(X, dtype=int)
        for feature in range(X.shape[1]):
            edges = np.linspace(X[:, feature].min(), X[:, feature].max(), self.num_bins + 1)
            binned_X[:, feature] = np.digitize(X[:, feature], edges) - 1  # Convert to bin indices
            bin_edges.append(edges)
        return binned_X, bin_edges

    def _grow_node(self, node, binned_X, gradients, hessians, depth, max_depth, n_jobs):
        if depth < max_depth and len(np.unique(gradients)) > 1:  # Check for stopping conditions
            best_feature, best_bin = self._find_best_split(binned_X, gradients, hessians, n_jobs)
            if best_feature is not None:
                node.value = best_feature
                node.threshold = self.bin_edges[best_feature][best_bin]  # Convert bin index to threshold
                node.is_leaf = False
                # Split the data
                left_indices = binned_X[:, best_feature] < best_bin
                right_indices = binned_X[:, best_feature] >= best_bin
                node.left = TreeNode()
                node.right = TreeNode()
                # Parallelize the growing of child nodes
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [
                        executor.submit(self._grow_node, node.left, binned_X[left_indices], gradients[left_indices], hessians[left_indices], depth + 1, max_depth, n_jobs),
                        executor.submit(self._grow_node, node.right, binned_X[right_indices], gradients[right_indices], hessians[right_indices], depth + 1, max_depth, n_jobs)
                    ]
                    for future in futures:
                        future.result()  # Wait for the child nodes to finish
        else:
            # Set the leaf prediction based on gradients and hessians
            sum_grad = np.sum(gradients)
            sum_hess = np.sum(hessians)
            node.prediction = sum_grad / (sum_hess + 1e-10)  # Add small value to avoid division by zero

    def _find_best_split(self, binned_X, gradients, hessians, n_jobs):
        best_feature = None
        best_bin = None
        best_gain = -float('inf')  # For regression, we maximize the gain
        
        # Parallelize the search for the best split
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for feature in range(binned_X.shape[1]):
                futures.append(executor.submit(self._find_split_for_feature, feature, binned_X, gradients, hessians))
            
            # Wait for all futures to complete and get the results
            results = [future.result() for future in futures]
            
            # Find the best split result from the computed results
            for res in results:
                if res['gain'] > best_gain:
                    best_gain = res['gain']
                    best_feature = res['feature']
                    best_bin = res['bin']

        return best_feature, best_bin

    def _find_split_for_feature(self, feature, binned_X, gradients, hessians):
        hist_grad = np.zeros(self.num_bins)
        hist_hess = np.zeros(self.num_bins)

        # Build histograms
        for bin_idx in range(self.num_bins):
            mask = binned_X[:, feature] == bin_idx
            hist_grad[bin_idx] = np.sum(gradients[mask]) # Sum of gradients for the bin
            hist_hess[bin_idx] = np.sum(hessians[mask]) # Sum of Hessians for the bin

        # Find the best split based on histograms
        best_gain = -float('inf')
        best_bin = None
        left_grad, left_hess = 0, 0
        total_grad, total_hess = np.sum(hist_grad), np.sum(hist_hess)

        for bin_idx in range(self.num_bins):
            left_grad += hist_grad[bin_idx] # Cumulative sum of gradients
            left_hess += hist_hess[bin_idx] # Cumulative sum of Hessians
            right_grad = total_grad - left_grad # Sum of gradients in the right child
            right_hess = total_hess - left_hess # Sum of Hessians in the right child

            if left_hess > 0 and right_hess > 0:  # Avoid division by zero
                gain = self._calculate_gain(left_grad, left_hess, right_grad, right_hess)
                if gain > best_gain:
                    best_gain = gain
                    best_bin = bin_idx

        return {'feature': feature, 'bin': best_bin, 'gain': best_gain}

    def _calculate_gain(self, left_grad, left_hess, right_grad, right_hess):
        """
            Formula: gain = 0.5 * (G_L^2 / (H_L + λ) + G_R^2 / (H_R + λ) - (G_L + G_R)^2 / (H_L + H_R + λ))
        """
        # Calculate the gain based on gradients and Hessians
        gain = 0.5 * (left_grad ** 2 / (left_hess + 1e-10) + 
                      right_grad ** 2 / (right_hess + 1e-10) - 
                      (left_grad + right_grad) ** 2 / (left_hess + right_hess + 1e-10))
        return gain

    def score(self, y, y_pred):
        return np.mean(y == y_pred)


