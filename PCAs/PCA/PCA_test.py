import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
class PCA:
    def __init__(self, threshold = .95):
        self.threshold = threshold
    def fit(self, X):
        X = StandardScaler().fit_transform(X)
        #1.Standardization
        X -= np.mean(X, axis = 0)
        #2.Covariance Matrix
        cov_mat = np.cov(X.T) #cov_mat = X.T @ X / (len(X) - 1)
        #3.Eigenvectors, Eigenvalues from Covariance Matrix
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        print('Eigenvectors:', eig_vecs)
        print('Eigenvalues:', eig_vals)
        print('SVD:', np.linalg.svd(cov_mat)[0])
        eig_vals = abs(eig_vals)
        #  3.1.Order Eigenvectors, Eigenvalues in descending order of Eigenvalues
        order = sorted(np.arange(len(eig_vals)), key = lambda x: -eig_vals[x])
        eig_vals, eig_vecs = eig_vals[order], eig_vecs[:, order]
        #4.Reduce dimensions from n demensions to k demensions
        percent = np.cumsum(eig_vals) / np.sum(eig_vals)
        x_axis = np.arange(len(eig_vals))
        plt.subplot(2,1, 1)
        plt.bar(x_axis, eig_vals / np.sum(eig_vals), label = 'Individuals')
        plt.xticks(x_axis, ['PC1', 'PC2', 'PC3', 'PC4'])
        plt.plot(x_axis, percent, marker = 'o', color = 'orange', label = 'Cumulative')
        plt.xlabel('Priciple Components')
        plt.ylabel('Explained Variance in Percent')
        plt.title('Contribution')
        plt.legend(loc = 5)
        k = np.argmax(percent >= self.threshold) + 1
        #5.Select k Eigenvectors, Eigenvalues with highest variance(highest Eigenvalues)

        self.basis = eig_vecs[:, :k]
        self.values = eig_vals[:k]
        print(self.basis, self.values)
        print(self.basis.shape, X.shape)
        return self.project(X)
    def project(self, X):
        return X @ self.basis


from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
clf = PCA(.9)
X = clf.fit(X)
colors = ['red', 'green', 'blue']
labels = ['Rose', 'Tulips', 'Blossom']

plt.subplot(2,1,2)
for x, y in zip(X, iris.target):
    plt.scatter(x[0], x[1], color = colors[y], alpha = .8)
plt.legend(iris.target_names)
plt.axis('equal')
plt.subplots_adjust(hspace = .5)
plt.show()
