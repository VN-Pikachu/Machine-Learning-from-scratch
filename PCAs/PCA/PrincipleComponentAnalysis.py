import numpy as np
#from sklearn.preprocessing import StandardScaler
#NOTE: v, u = numpy.linalg.eig(matrix)
#u is matrix having columns as eigenvectors
#v is a vector of Eigenvalues respect to each column of eigenvectors of u
class PCA:
    def __init__(self, n_components = None, threshold = .95):
        self.threshold = threshold
        self.n_components = n_components
    def fit(self, X):
        #X = StandardScaler().fit_transform(X)
        #1.Standardization
        self.mean_ = np.mean(X, axis = 0)
        X -= self.mean_
        #2.Covariance Matrix
        cov_mat = np.cov(X.T) #cov_mat = X.T @ X / (len(X) - 1)
        #3.Eigenvectors, Eigenvalues from Covariance Matrix
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        eig_vals = abs(eig_vals)
        #  3.1.Order Eigenvectors, Eigenvalues in descending order of Eigenvalues
        order = sorted(np.arange(len(eig_vals)), key = lambda x: -eig_vals[x])
        eig_vals, eig_vecs = eig_vals[order], eig_vecs[:, order]
        #4.Reduce dimensions from n demensions to k demensions
        percent = np.cumsum(eig_vals) / np.sum(eig_vals)
        #Chose k demensions so that the explained variance ratio >= threshold
        k = np.argmax(percent >= self.threshold) + 1
        #If setting n_components, not threshold
        if self.n_components is not None:
            k = self.n_components
        '''Equivelent to:
        total_vals = np.sum(eig_vals)
        k = tmp = 0
        while tmp <= self.threshold:
            tmp += eig_vals[k] / total_vals
            k += 1
        '''
        #5.Select k Eigenvectors, Eigenvalues with highest variance(highest Eigenvalues)
        eig_vecs, eig_vals = eig_vecs[:, :k], eig_vals[:k]
        self.basis = eig_vecs
        #Return a matrix of eigenvectors, each row is an eigenvector
        self.components_ = eig_vecs.T
        self.explained_variance_ = eig_vals
        self.explained_variance_ratio_ = eig_vals / np.sum(eig_vals)
        return self.transform(X)
    def transform(self, X):
        return X @ self.basis
    def inverse_transform(self, X):
        return X @ self.basis.T + self.mean_
