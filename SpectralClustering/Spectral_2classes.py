import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csgraph
def Gaussian_Kernel(x1, x2, gamma = 1):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * gamma ** 2))

class SpectralClustering:
    def __init__(self, n_clusters = None, gamma = 1):
        self.n_clusters_ = n_clusters
        self.gamma = gamma
    def fit(self, X):
        n, m = X.shape #n samples, m features
        #Affinity Matrix
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                A[i,j] = A[j, i] = Gaussian_Kernel(X[i], X[j], self.gamma)
        #Degree Matrix
        D = np.zeros((n, n))
        for i in range(n):
            D[i,i] = np.sum(A[i])
        #Laplacian Matrix
        L = D - A
        #Eigenvectors, eigenvalues of the Laplacian Matrix L ordered by eigenvalues ascending
        eig_vals, eig_vecs = np.linalg.eig(L)
        order = sorted(np.arange(len(eig_vals)), key = lambda x: eig_vals[x])
        eig_vals = eig_vals[order]
        eig_vecs = eig_vecs[:,order]

        model = KMeans(n_clusters = 2)
        model.fit(eig_vecs[:, 1].reshape((-1, 1)))
        self.labels_ = model.labels_

from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
X,y = make_moons(150, noise = .07, random_state = 21)
#X, y = make_circles(150, noise = .05, factor = .5)
clf = SpectralClustering(n_clusters = 2, gamma = .1)
clf.fit(X)
plt.scatter(X[:,0], X[:,1], c = clf.labels_, cmap = 'RdBu')
plt.show()
