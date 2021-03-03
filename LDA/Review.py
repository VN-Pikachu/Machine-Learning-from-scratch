import numpy as np
class LDA:
    def __init__(self, n_components = None):
        self.n_components_ = n_components
    def fit(self, X, y):
        M = np.mean(X, axis = 0)
        n, m = X.shape
        S_B = np.zeros((m, m))
        S_W = np.zeros((m, m))
        for group in np.unique(y):
            x = X[y == group]
            Mi = np.mean(x, axis = 0)
            x -= Mi
            Ni = len(x)
            S_B += Ni * np.outer(Mi - M, Mi - M)
            S_W += x.T @ x

        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W) @ S_B)
        keys = sorted(range(len(eig_vals)), key = lambda x : -abs(eig_vals[x]))
        eig_vecs = eig_vecs[:, keys]
        eig_vals = eig_vals[keys]

        if self.n_components_ == None:
            self.n_components_ = m

        self.components_ = eig_vecs[:, :self.n_components_].T
        self.explained_variance_ = eig_vals[:self.n_components_]
        self.explained_variance_ratio = np.cumsum(eig_vals) / np.sum(eig_vals)

    def transform(self, X):
        return X @ self.components_.T

samples = np.array([
[1,2],
[2,3],
[3,3],
[4,5],
[5,5],
[4,2],
[5,0],
[5,2],
[3,2],
[5,3],
[6,3]
]).astype('float64')
labels = np.array([0] * 5 + [1] * 6)

model = LDA()
model.fit(samples, labels)

import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.datasets import load_iris
import math
iris = load_iris()
fig, axes = plt.subplots(2,2,figsize = (10, 5))
fig.suptitle('Iris Feature Distribution')
for i, ax in enumerate(axes.flat):
    bins = np.linspace(math.floor(min(iris.data[:, i])), math.ceil(max(iris.data[:, i])), 25)
    for j, color in zip(range(3), ['red', 'green', 'blue']):
        ax.hist(iris.data[iris.target == j,i], bins = bins, color = color, alpha = .6,
        label = iris.target_names[j])
        ax.tick_params(axis = 'both', top = 'off', bottom = 'off', left = 'off', right = 'off',
        labelleft = 'on', labelbottom = 'on')
    for position in ['top', 'left', 'bottom', 'right']:
        ax.spines[position].set_visible(False)
    leg = ax.legend(loc = 1, fancybox = True, fontsize = 8)
    leg.get_frame().set_alpha(.5)
    ax.set_xlabel(iris.feature_names[i])

axes[0,0].set_ylabel('count')
axes[1,0].set_ylabel('count')

fig.tight_layout()
plt.show()
