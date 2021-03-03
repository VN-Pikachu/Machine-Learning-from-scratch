import numpy as np
from Affinity import *
from sklearn.cluster import KMeans

class SpectralClustering:
    def __init__(self, n_clusters = 8, affinity = 'rbf', assign_label = 'kmeans', gamma = 1, n_neighbors = 10):
        self.n_clusters_ = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors

    def fit(self, X):
        n, m = X.shape
        #Build the similariy matrix
        if self.affinity == 'rbf':
            A = RBF(X, self.gamma)
        if self.affinity == 'nearest_neighbors':
            A = KNearestNeighbors(X, n_neighbors = self.n_neighbors)
        #Degree Matrix (Normalized)
        D = np.diag(np.sum(A, axis = 1) ** -.5)

        #Eigenvectors, Eigenvalues ordered by eigenvalues descending
        eig_vals, eig_vecs = np.linalg.eig(D @ A @ D)
        #order = sorted(range(len(eig_vals)), key = lambda x: -eig_vals[x])
        order = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[order]
        eig_vecs = eig_vecs[:, order]
        #Chose k largest eigenvectors
        U = eig_vecs[:, :self.n_clusters_]
        #Normalize each row of U to unit length
        U /= np.linalg.norm(U, axis = 1).reshape((-1,1))
        #Run clustering algorithmn on embedding space U
        model = KMeans(n_clusters = self.n_clusters_)
        model.fit(U)
        self.labels_ = model.labels_
