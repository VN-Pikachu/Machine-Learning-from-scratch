import numpy as np
class DPCA:
    def __init__(self,n_components = 2):
        self.n_components = n_components
    def fit(self, X):
        self.mean = np.mean(X, axis = 0)
        X -= np.mean(X, axis = 0)
        V, M2, VT =  np.linalg.svd(X @ X.T)
        self.components = (np.diag(1 / np.sqrt(M2)) @ VT @ X)[:self.n_components]
        return self.transform(X)
    def transform(self, X):
        return X @ self.components.T
    def inverse_transform(self, X):
        return X @ self.components + self.mean
