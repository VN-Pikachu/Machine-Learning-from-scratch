#todo:https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.neighbors import kneighbors_graph
ax = plt.subplot(121, projection = '3d')
data, labels = make_swiss_roll(1500)
model = AC(n_clusters = 6)
y = model.fit_predict(data)
ax.scatter(data[:,0], data[:,1], data[:,2],c = y, cmap = 'rainbow', s = 10, edgecolor = 'k')
ax.set_title('Without connectivity constraint')
ax.axis('off')


ax = plt.subplot(122, projection = '3d')
model = AC(n_clusters = 6, connectivity = kneighbors_graph(data, n_neighbors = 10))
y = model.fit_predict(data)
ax.scatter(data[:,0], data[:,1], data[:,2], c = y, cmap = 'rainbow', s = 10, edgecolor = 'k')
ax.set_title('With connectivity constraint')
ax.axis('off')

plt.show()
