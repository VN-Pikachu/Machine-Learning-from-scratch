import numpy as np
from collections import defaultdict
from numpy.linalg import norm
class KMeansClassifier:
    def __init__(self, n_clusters = 2, tol = 0.0001, max_iter = 300):
        self.max_iter = max_iter
        self.tol = tol
        self.n_clusters = n_clusters
        self.labels_ = None
    def fit(self, X):
        self.cluster_centers_ = X[np.random.choice(len(X), size = self.n_clusters, replace = False)]
        cluster = defaultdict(list)

        for _ in range(self.max_iter):
            #assign each input data x to a cluster (whose centroid is closest to x)
            labels_ = [np.argmin(norm(x - self.cluster_centers_, axis = 1)) for x in X]
            if self.labels_ == labels_: break #Converged
            self.labels_ = labels_
            #Group input data into it's cluster
            for i, cluster_i in enumerate(self.labels_):
                cluster[cluster_i].append(X[i])
            #Move centroid of each cluster
            for i in cluster:
                self.cluster_centers_[i] = np.mean(cluster[i], axis = 0)
