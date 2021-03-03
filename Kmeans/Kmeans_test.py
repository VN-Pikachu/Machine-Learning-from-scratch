from Kmeans import KMeansClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X = np.concatenate((X0, X1, X2), axis = 0)

colors = ['r', 'g', 'b']
markers = ['^', 'o', 's']
clf = KMeansClassifier(n_clusters = 3)
#clf = KMeans(n_clusters = 3, random_state = 0)
clf.fit(X)
centroids = clf.cluster_centers_
for p, i in zip(X, clf.labels_):
    plt.scatter(p[0], p[1], c = colors[i], marker = markers[i], alpha = .8, s = 25)
plt.scatter(centroids[:, 0], centroids[:, 1], c = 'black', marker = 'x', s = 40)
plt.show()
