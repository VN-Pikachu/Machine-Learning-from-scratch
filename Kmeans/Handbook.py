import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

X = np.array(make_blobs(n_samples = 300, centers = 4, cluster_std = .6, random_state = 0)[0])
clf = KMeans(n_clusters = 4)
clf.fit(X)
centroids, labels = clf.cluster_centers_, clf.labels_
plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = 'rainbow', s = 25)
plt.scatter(centroids[:, 0], centroids[:, 1], c = 'b', s = 50, alpha = .3)
plt.axis('tight')
plt.show()


from sklearn.datasets import make_moons
X = make_moons(200, noise = .05, random_state = 0)[0]
model = KMeans(n_clusters = 2)
labels = model.fit_predict(X)
plt.scatter(X[:,0], X[:, 1], c = labels, cmap = 'spring', s = 25)
plt.show()

from sklearn.datasets import load_digits
data = load_digits()
digits = data.data
model = KMeans(n_clusters = 10, random_state = 0)
labels = model.fit_predict(digits)
centroids = model.cluster_centers_.reshape((10, 8, 8))
fig, axes = plt.subplots(2, 5, figsize = (8, 3))
for ax, num in zip(axes.flat, centroids):
    ax.set(xticks = [], yticks = [])
    ax.imshow(num, cmap = 'binary')
plt.show()
