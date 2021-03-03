import numpy as np
from sklearn.neighbors import kneighbors_graph
class LLE:
    def __init__(self, n_components = 2, n_neighbors = 5, alpha = 0.001):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.alpha = alpha
    def fit(self, X):
        n, p = X.shape
        K = kneighbors_graph(X, n_neighbors = self.n_neighbors).toarray().astype('bool')
        W = np.zeros((n,n))
        for i in range(n):
            M = X[K[i]] - X[i]
            G = M @ M.T
            w = .5 * np.linalg.inv(G + self.alpha * np.eye(self.n_neighbors)) @ np.ones(self.n_neighbors)
            W[i, K[i]] = w / np.sum(w)
        I = np.eye(n)
        L = (I - W).T @ (I - W)
        vals, vecs = np.linalg.eigh(L)
        Y = vecs[:, 1:self.n_components + 1]
        self.embedding_ = Y
        self.reconstruction_error_ = np.trace(Y.T @ L @ Y)
        return Y
