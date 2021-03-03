import numpy as np

class CMDS:
    def __init__(self, n_components = 2):
        self.n_components = n_components
    def fit(self, D):
        n = len(D)
        H = np.eye(n) - np.ones((n, n)) / n
        G = -.5 * H @ (D ** 2) @ H
        U, S, V = np.linalg.svd(G)
        X = U @ np.diag(S ** .5)
        self.embedding_ = X[:, :self.n_components]
        return self.embedding_
