import numpy as np
import matplotlib.pyplot as plt
class LDA:
    def __init__(self, n_components = None):
        self.n_components_ = n_components
    def fit(self, X, y):
        #n samples, m features
        n, m = X.shape
        #Between-Class Variance Matrix
        S_B = np.zeros((m,m))
        #Within-Class Variance Matrix
        S_W = np.zeros((m,m))
        #Global mean over all samples
        M = np.mean(X, axis = 0)
        #Go though each class
        for group in np.unique(y):
            #Get samples from the current class
            x = X[y == group]
            #Local mean of the current group
            Mi = np.mean(x, axis = 0)
            #Centering-data
            x -= Mi
            #The number of samples of the current group
            Ni = len(x)
            #Update Between-Class Variance Matrix and Within-Class Variance Matrix
            S_B += Ni * np.outer(Mi - M, Mi - M)
            S_W += x.T @ x #Covariance Matrix: np.cov(x.T) : normalized

        #Calculate Eigenvectors, Eigenvalues of: inv(S_W) @ S_b
        #Order by eigenvalues descending
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W) @ S_B)
        keys = sorted(range(len(eig_vals)), key = lambda x: -abs(eig_vals[x]))
        eig_vecs = eig_vecs[:, keys]
        eig_vals = eig_vals[keys]

        if self.n_components_ == None:
            self.n_components_ = m

        self.components_ = eig_vecs[:, :self.n_components_].T
        self.explained_variance_ = eig_vals[:self.n_components_]
        self.explained_variance_ratio_ = [eig_vals / np.sum(eig_vals)][:self.n_components_]

    def transform(self, X):
        return X @ self.components_.T
