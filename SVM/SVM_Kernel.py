#NOTE: Reference:https://pythonprogramming.net/soft-margin-kernel-cvxopt-svm-machine-learning-tutorial/
import numpy as np
from numpy import linalg
import cvxopt
from cvxopt import matrix
def linear_kernel(x, y):
    return x @ y
def polynomial_kernel(x, y, p = 3):
    return (x @ y + 1) ** p
def gaussian_kernel(x, y, gamma = 1):
    return linalg.norm(x - y) ** 2 / (2 * gamma ** 2)
class SVM:
    def __init__(self, kernel = linear_kernel, C = None):
        self.kernel = kernel
        self.C = C
    def fit(self, X, y):
        n_samples, n_features = X.shape
        #Gram Matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        Q = matrix(np.outer(y, y) * K)
        p = matrix(np.ones(n_samples) * -1)
        G = matrix(np.diag(np.ones(n_samples) * -1))
        h = matrix(np.zeros(n_samples))
        A = matrix(y.astype('double'), (1,n_samples))
        b = matrix(0.0)
        #Soft Margin
        if self.C is not None:
            G = matrix(np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples))))
            h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        a = np.ravel(cvxopt.solvers.qp(Q, p, G, h, A, b)['x'])
        sv = a > 1e-5
        self.a, self.X, self.y, self.K = a[sv], X[sv], y[sv], K[sv][:, sv]
        #Bias
        self.b = np.sum(self.y - self.a * self.y @ self.K) / len(self.a)
        #Weight
        self.w = None
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for a, y, X in zip(self.a, self.y, self.X):
                self.w += a * y * X

    def project(self, X):
        if self.kernel == linear_kernel:
            return X @ self.w + self.b
        return np.array([sum(a * y * self.kernel(x, u) for a, y, x in zip(self.a, self.y, self.X)) + self.b for u in X])

    def predict(self, X):
        return np.sign(self.project(X))
