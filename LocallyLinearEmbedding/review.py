import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
class LLE:
    def __init__(self, n_components = 2, n_neighbors = 5, alpha = 0.001):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.alpha = alpha
    def fit(self, X):
        k = self.n_neighbors
        M = kneighbors_graph(X, n_neighbors = k, mode = 'connectivity').toarray().astype('bool')
        n = len(X)
        W = np.zeros((n,n))
        for i in range(n):
            x = X[M[i]] - X[i]
            G = x @ x.T + self.alpha * np.eye(k)
            wi = np.linalg.inv(G) @ np.ones(k)
            W[i, M[i]] = wi / np.sum(wi)

        I = np.eye(n)
        L = (I - W).T @ (I - W)
        Y = np.linalg.eigh(L)[1][:, 1:1 + self.n_components]
        self.embedding_ = Y
        self.reconstruction_error_ = Y.T @ L @ Y
        return self.embedding_, self.reconstruction_error_

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns;sns.set()

data, labels = make_swiss_roll(1500, random_state = 6)
model = LLE(2, 20, 0.1)
X, error_ = model.fit(data)
fig = plt.figure()
ax = fig.add_subplot(211, projection = '3d')
ax.scatter(data[:,0], data[:,1], data[:,2], s = 20, c = labels, cmap = 'Spectral')
ax.set_title('Orginal Data')
ax = fig.add_subplot(212)
ax.scatter(X[:,0], X[:,1],s = 20, c = labels, cmap = 'Spectral')
ax.set_title('transform data')

plt.show()
