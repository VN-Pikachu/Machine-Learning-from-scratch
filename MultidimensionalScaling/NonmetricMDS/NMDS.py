import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.spatial import distance_matrix
from numpy.linalg import norm
from itertools import combinations
class NMDS:
    def __init__(self, n_components = 2, max_iter = 300, alpha = .2, tolerate = 1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.alpha = alpha
        self.tolerate = tolerate
    def fit(self, X):
        N = len(X)
        distance = distance_matrix(X, X)

        order = list(combinations(range(N),2))
        order.sort(key = lambda pair:distance[pair[0], pair[1]])
        order = {pair:i for i, pair in enumerate(order)}
        STRESS_1 = float('inf')
        M = 100 * np.random.rand(N,self.n_components)

        for _ in range(self.max_iter):
            if STRESS_1 <= self.tolerate: break
            d = [norm(M[i] - M[j]) for i, j in order.keys()]
            d_ = IsotonicRegression().fit_transform(range(len(d)), d)
            #Gradient Descent Minimize raw stress
            Gradient = np.zeros(M.shape)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        pair = (i,j) if i < j else (j,i)
                        k = order[pair]
                        Gradient[i] += (1 - d_[k] / d[k]) * (M[i] - M[j])
            M -= self.alpha / (N - 1) * Gradient
            STRESS_1 = (norm(d - d_)** 2 / norm(d) ** 2) ** .5

        self.embedding_ = M
        self.stress_ = STRESS_1
        return self.embedding_, self.stress_
