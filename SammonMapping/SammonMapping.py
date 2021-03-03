import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
from itertools import combinations
class SammonMapping:
    def __init__(self, n_components = 2, max_iter = 300, alpha = .2):
        self.max_iter = max_iter
        self.n_components = n_components
        self.alpha = alpha
    def fit(self, X):
        N = len(X)
        D = distance_matrix(X, X)
        M = 100 * np.random.rand(N,self.n_components)

        for _ in range(self.max_iter):
            d = distance_matrix(M, M)

            Gradient = np.zeros(M.shape)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        Gradient[i] += (1 - D[i,j] / d[i,j]) * (M[i] - M[j])
            M -= self.alpha / (N - 1) * Gradient
        self.stress_ = 1 / np.sum(D) * sum((D[i,j] - np.linalg.norm(M[i] - M[j])) ** 2 / D[i,j] for i, j in combinations(range(N), 2))
        self.embedding_ = M
        return self.embedding_
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.datasets import load_iris
iris = load_iris()
model = SammonMapping(n_components = 2, max_iter = 90)

X = model.fit(iris.data)
plt.scatter(X[:,0], X[:,1], c = iris.target, cmap = 'spring')
plt.show()
