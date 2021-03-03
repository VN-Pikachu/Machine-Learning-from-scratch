import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans

class GMM:
    def __init__(self, n_components, n_iter = 100):
        self.n_components = n_components
        self.n_iter = n_iter
    def fit(self, X):
        N, d = X.shape
        model = KMeans(n_clusters = self.n_components)
        model.fit(X)
        self.means_ = model.cluster_centers_
        self.covars_ = np.zeros((N,d,d))
        self.weights_ = np.zeros(self.n_components)
        for k in range(self.n_components):
            elements = X[model.labels_ == k]
            self.weights_[k] = len(elements) / N
            self.covars_[k] = np.cov(elements.T)
        for _ in range(self.n_iter):
            P = np.zeros((self.n_components, N))
            for k in range(self.n_components):
                P[k] = self.weights_[k] * mvn.pdf(X, self.means_[k], self.covars_[k])
            P /= np.sum(P, axis = 0)
            means = np.zeros_like(self.means_)
            covars = np.zeros_like(self.covars_)
            self.weights_ = np.mean(P, axis = 1)
            for k in range(self.n_components):
                means[k] = np.sum(X * P[k].reshape((-1,1)), axis = 0)
                for i, x in enumerate(X):
                    covars[k] += P[k][i] * np.outer(x - self.means_[k], x - self.means_[k])
                means[k] /= np.sum(P[k])
                covars[k] /= np.sum(P[k])
            self.means_ = means
            self.covars_ = covars

        self.labels_ = self.predict(X)

    def predict_proba(self, X):
        N = len(X)
        ans = np.zeros((N, self.n_components))
        for k in range(self.n_components):
            ans[:,k] = self.weights_[k] * mvn.pdf(X, self.means_[k], self.covars_[k])
        return ans
    def predict(self, X):
        return np.argmax(self.predict_proba(X),axis = 1)


import matplotlib.pyplot as plt;
import seaborn as sns;sns.set()
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
digits = load_digits()
data = pca.fit_transform(digits.data)

model = GMM(n_components = 10)
model.fit(data)
plt.scatter(data[:,0], data[:,1], c = model.labels_, cmap = 'spring')
plt.show()
