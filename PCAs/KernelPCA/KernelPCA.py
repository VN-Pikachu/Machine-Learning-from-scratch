import numpy as np
def RBF(x, y, bandwidth):
    return np.exp(- np.linalg.norm(x - y) ** 2 / (2 * bandwidth ** 2))
def Polynomial(x, y, p): #(x.y + 1) ** p
    return (x @ y + 1) ** p
def Linear(x, y):
    return x @ y

#kernel choices : 'rbf', 'linear', 'polynomial'
#degree : only apply for Polynomial kernel, equivalent to parameter p in the Polynomial function
#badwidth: only apply for RBF kernel, equivalent to parameter bandwidth
class KPCA:
    def __init__(self, n_components = 2, kernel = 'rbf', degree = 2, bandwidth = 1):
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.bandwidth = bandwidth
        self.CenterMatrix = None
    def KernelMatrix(self, X, Y):
        N, M = len(X), len(Y)
        K = np.zeros((N,M))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i,j] = RBF(x, y, self.bandwidth) if self.kernel == 'rbf' else \
                Polynomial(x, y, self.degree) if self.kernel == 'polynomial' else \
                Linear(x, y)
        return K

    def fit(self, X):
        self.X = X
        N = len(X)
        H = np.eye(N) - np.ones((N,N)) / N
        K = H @ self.KernelMatrix(X, X) @ H #Center Kernel Matrix
        V, M2, VT = np.linalg.svd(K)
        M = np.diag(np.sqrt(M2))
        self.V = V[:, :self.n_components]
        self.M = M[:self.n_components, :self.n_components]
        return self.V @ self.M
    def transform(self, data):
        return self.KernelMatrix(self.X, data).T @ self.V @ np.linalg.inv(self.M)
