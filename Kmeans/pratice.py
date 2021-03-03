import numpy as np
from numpy.linalg import norm
from collections import defaultdict
import matplotlib.pyplot as plt
class KMeans:
    def __init__(self, n_clusters, tol = 0.0001, max_iter = 300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
    def fit(self, X):
        self.cluster_centers_ = X[np.random.choice(len(X), size = self.n_clusters, replace = False)]
        cluster = defaultdict(list)
        for _ in range(self.max_iter):
            self.labels_ = [np.argmin(norm(x - self.cluster_centers_, axis = 1)) for x in X]
            for i, i_cluster in enumerate(self.labels_):
                cluster[i_cluster].append(X[i])
            #next_centroids = np.zeros(self.n_clusters)
            for i in cluster:
                self.cluster_centers_[i] = np.mean(cluster[i], axis = 0)
                #next_centroids[i] = np.mean(cluster[i], axis = 0)
            #if np.sum(abs(next_centroids - self.cluster_centers_)) <= tol: break
            #self.cluster_centers_ = next_centroids

N = 500
mean = [[2, 2], [8, 3], [3, 6]]
cov = np.identity(2)
X1 = np.random.multivariate_normal(mean[0], cov, N)
X2 = np.random.multivariate_normal(mean[1], cov, N)
X3 = np.random.multivariate_normal(mean[2], cov, N)
X = np.concatenate((X1, X2, X3), axis = 0)

clf = KMeans(n_clusters = 3)
clf.fit(X)
colors = ['r', 'g', 'b']
markers = ['^', 's', 'o']
for x, label in zip(X, clf.labels_):
    plt.scatter(x[0], x[1], c = colors[label], s = 20, marker = markers[label])
centroids = clf.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 40, c = 'teal')
plt.show()
