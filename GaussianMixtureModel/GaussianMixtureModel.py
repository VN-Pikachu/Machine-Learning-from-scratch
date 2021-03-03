import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans

class GMM:
    def __init__(self, n_components = 1, n_iter = 100):
        self.n_components = n_components
        self.n_iter = n_iter
    def fit(self, X):
        N, d = X.shape
        model = KMeans(n_clusters = self.n_components)
        model.fit(X)
        self.means_ = model.cluster_centers_
        self.covars_ = np.zeros((self.n_components, d,d))
        self.weights_ = np.zeros(self.n_components)
        for k in range(self.n_components):
            elements = X[model.labels_ == k]
            Nk = len(elements)
            self.weights_[k] = Nk / N
            self.covars_[k] = np.cov(elements.T)

        for _ in range(self.n_iter):
            '''' ---------------------- E-step --------------------------------'''
            #Posterior Distribution P(k|X) ---> P[k][i]: P(k|Xi) = P(Xi|k) * P(k) / P(Xi)
            P = np.zeros((self.n_components, N))
            #Palculate P(Xi|k) * P(k) fo each cell (k, i)
            for k in range(self.n_components):
                P[k] = mvn.pdf(X, self.means_[k], self.covars_[k])
            #Divide each column by P(Xi)
            print('Log Likelihood:', np.sum(np.log(np.sum(P,axis = 0))))
            P /= np.sum(P, axis = 0)
            ''' ---------------------- M-step ----------------------------------'''
            means = np.zeros_like(self.means_)
            covars = np.zeros_like(self.covars_)
            for k in range(self.n_components):
                for i in range(N):
                    means[k] += P[k][i] * X[i]
                    covars[k] += P[k][i] * np.outer(X[i] - self.means_[k], X[i] - self.means_[k])
                means[k] /= np.sum(P[k])
                covars[k] /= np.sum(P[k])
            self.means_ = means
            self.covars_ = covars
            self.weights_ = np.mean(P, axis = 1) #self.weights_[k] = Nk / N

        self.labels_ = self.predict(X)
        return self.labels_

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis = 1)
    def predict_proba(self, X):
        #P[i][k] : P(Xi|k) * P(k)
        P = np.zeros((len(X), self.n_components))
        for k in range(self.n_components):
            P[:,k] = self.weights_[k] * mvn.pdf(X, self.means_[k], self.covars_[k])
        return P
