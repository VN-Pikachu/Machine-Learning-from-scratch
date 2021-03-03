import numpy as np
def Linear(x, y): return x @ y
def RBF(x, y, bandwidth): return np.exp(-  np.linalg.norm(x - y) ** 2 / (2 * bandwidth ** 2))
def Polynomial(x, y, degree): return (x @ y + 1) ** degree

class KSPCA:
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
        self.X = X
        N = len(X)
        H = np.eye(N) - 1 / N * np.ones((N,N))
        B = self.KernelMatrix(Y,Y)
        K = self.KernelMatrix(X,X)
        eig_vals, eig_vecs = np.linalg.eigh(H @ B @ H @ K)
        self.beta_ = np.fliplr(eig_vecs)[:,:self.n_components]
        return K.T @ self.beta_
    def transform(self, data):
        return self.KernelMatrix(self.X, data).T @ self.beta_
