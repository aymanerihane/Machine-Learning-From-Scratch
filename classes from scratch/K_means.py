import numpy as np

class KMeans:

    def __init__(self, max_iters=100, k = 3):
        self.max_iters = max_iters
        self.k = k

    def initialize_centroids(self,X, K):
        # Randomly select K data points as initial centroids
        random_indices = np.random.choice(X.shape[0], K, replace=False)
        centroids = X[random_indices]
        return centroids

    def assign_clusters(self,X, centroids):
        # Assign each data point to the nearest centroid
        clusters = []
        for x in X:
            distances = [np.linalg.norm(x - centroid) for centroid in centroids]
            cluster = np.argmin(distances)  # Find the index of the nearest centroid
            clusters.append(cluster)
        return np.array(clusters)

    def update_centroids(self,X, clusters, K):
        # Calculate new centroids as the mean of all points in each cluster
        new_centroids = []
        for k in range(K):
            cluster_points = X[clusters == k]
            new_centroid = cluster_points.mean(axis=0) if len(cluster_points) > 0 else np.random.randn(X.shape[1])
            new_centroids.append(new_centroid)
        return np.array(new_centroids)

    def fit(self,X):
        # Step 1: Initialize centroids
        centroids = self.initialize_centroids(X, self.K)
        for i in range(self.max_iters):
            # Step 2: Assign points to the nearest centroid
            clusters = self.assign_clusters(X, centroids)
            
            # Step 3: Update centroids based on the current cluster assignments
            new_centroids = self.update_centroids(X, clusters, self.K)
            
            # Check for convergence (if centroids do not change)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        return centroids, clusters


    def error_rate(self,y_true, y_pred):
        incorrect = np.sum(y_true != y_pred)  # Count incorrect predictions
        print(incorrect)
        total = len(y_true)
        print(total)                   # Total predictions
        return incorrect / total  