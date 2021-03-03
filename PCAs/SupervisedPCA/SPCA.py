import numpy as np
def Linear(x, y): return x @ y
def RBF(x, y, bandwidth): return np.exp(-  np.linalg.norm(x - y) ** 2 / (2 * bandwidth ** 2))
def Polynomial(x, y, degree): return (x @ y + 1) ** degree
class SPCA:
    def __init__(self, n_components = 2, kernel = 'rbf', bandwidth = 1, degree = 3):
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.bandwidth = bandwidth
    def KernelMatrix(self, X, Y):
        N, M = len(X), len(Y)
        K = np.zeros((N,M))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i,j] = RBF(x,y,self.bandwidth) if self.kernel == 'rbf' else \
                Polynomial(x,y,self.degree) if self.kernel == 'Polynomial' else \
                Linear(x, y)
        return K
    def fit(self, X, Y):
        N = len(X)
        H = np.eye(N) - 1 / N * np.ones((N,N))
        B = self.KernelMatrix(Y, Y)
        G = X.T @ H @ B @ H @ X
        eig_vals, eig_vecs = np.linalg.eigh(G) #eigenvalues ascending order
        self.components_ = np.fliplr(eig_vecs)[:,:self.n_components] # Columns as Eigenvectors
        return X @ self.components_
    def transform(self, X):
        return X @ self.components_
    def inverse_transform(self, X):
        return X @ self.components_ @ self.components_.T
