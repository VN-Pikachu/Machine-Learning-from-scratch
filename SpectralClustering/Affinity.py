import numpy as np

def Gaussian_Kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * gamma ** 2))

def KNearestNeighbors(X, n_neighbors):
    n, m = X.shape
    A = np.zeros((n, n))
    for i in range(n):
        data = []
        for j in range(n):
            if i != j:
                data.append((np.linalg.norm(X[i] - X[j]), j))
        data.sort()
        for w, j in data[:n_neighbors]:
            A[i, j] = w
    return A

def RBF(X, gamma):
    n, m = X.shape
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(i + 1, n):
            A[i,j] = A[j,i] = Gaussian_Kernel(X[i], X[j], gamma)
    return A
