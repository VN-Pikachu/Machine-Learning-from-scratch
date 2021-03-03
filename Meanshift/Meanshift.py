import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
def gaussian_kernel(t):
    return np.exp(-t / 2)
class Meanshift:
    def __init__(self, kernel = gaussian_kernel, h = .5, max_iter = 100):
        self.kernel = kernel
        self.max_iter = max_iter
        self.h = h
    def fit(self, X):
        for _ in range(self.max_iter):
            for i, x in enumerate(X):
                weight = self.kernel(norm(x - X, axis = 1) ** 2 / self.h ** 2)
                X[i] = np.sum(weight.reshape((-1, 1)) * X, axis = 0) / np.sum(weight)
        self.cluster_centers_ = X
