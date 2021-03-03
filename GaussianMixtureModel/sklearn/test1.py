from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt;
import numpy as np
import seaborn as sns;sns.set();
data, labels = make_blobs(400, cluster_std = .6, random_state = 0, centers = 4)
data = data @ [[0.2,0.6], [0.5,.4]]
model = KMeans(n_clusters = 4)
y = model.fit_predict(data)
for k, center in enumerate(model.cluster_centers_):
    r = np.max(cdist([center], data[model.labels_ == k]))
    ax = plt.gca()
    ax.add_patch(plt.Circle(center, r, fc = '#CCCCCC', lw = 1, ec = 'teal', zorder = 1, alpha = .5))
plt.scatter(data[:,0], data[:,1], c = y, cmap = 'rainbow', zorder = 2, edgecolor = 'k', s = 20)
plt.show()
'''
model = KMeans(n_clusters = 4)
y = model.fit_predict(data)
for k, center in enumerate(model.cluster_centers_):
    X = data[model.labels_ == k]
    r = np.max(cdist(X, [center]))
    ax = plt.gca()
    ax.add_patch(plt.Circle(center , r, fc = '#CCCCCC', lw = .5, ec = 'teal', alpha = .5, zorder = 1))
plt.scatter(data[:,0], data[:,1], c = y, cmap = 'rainbow', s = 20, edgecolor = 'k', zorder = 2)
plt.axis('equal')
plt.show()
'''
