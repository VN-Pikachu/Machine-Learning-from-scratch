from HierarchicalClustering import *
from sklearn.datasets import make_moons, make_circles, make_blobs
import matplotlib.pyplot as plt;
import seaborn as sns;sns.set();

''' Make circles --------------------------------'''
'''
data, labels = make_circles(100, noise = .05, random_state = 7, factor = .5)
linkage = 'ward'
model = AgglomerativeClustering(2, linkage = linkage)
model.fit(data)
plt.scatter(data[:,0], data[:,1], c = model.labels_, cmap = 'rainbow')
plt.xlabel('linkage: %s' % linkage)
plt.show()
'''
data, labels = make_blobs(1500, cluster_std = [1.0, 2.5, 0.5], random_state = 170)
l = 'complete'
model = AgglomerativeClustering(2, linkage = l)
model.fit(data)
plt.scatter(data[:,0], data[:,1], c = model.labels_, cmap = 'rainbow')
plt.xlabel('likange:%s' % l)
plt.show()
