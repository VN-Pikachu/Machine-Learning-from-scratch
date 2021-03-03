#todo: https://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html#sphx-glr-auto-examples-cluster-plot-digits-linkage-py
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.datasets import make_blobs, make_moons, make_circles, load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt;
from sklearn.preprocessing import StandardScaler
import time
import numpy as np

#import seaborn as sns;sns.set()
np.random.seed(0)
N = 1500
random_state = 170
moons = make_moons(N, noise = .05, random_state = random_state)
circles = make_circles(N, factor = .5, noise = .05, random_state = random_state)
blobs = make_blobs(N, centers = 3, random_state = random_state)
X, y = make_blobs(N, centers = 3, random_state = random_state)
X = X @ [[0.6, -0.6], [-0.4, 0.8]]
skew = X, y
square = np.random.rand(N, 2), None
blow = make_blobs(N, cluster_std = [1., 2.5, 0.5], random_state = random_state)
datasets = (moons, circles, blobs, blow, skew, square)
linkages = ['ward', 'complete', 'average', 'ward']
fig, axes = plt.subplots(6,4,figsize = (9 * 1.3 + 2, 14.5), subplot_kw={'xticks':[], 'yticks':[]},
 gridspec_kw={'left':0.02,'right':0.98, 'wspace':0.05, 'hspace':0.01, 'top':0.96, 'bottom':0.001})
t0 = time.time()
fig.suptitle('Agglomerative Clustering')
clusters = [2] * 2 + [3] * 4
for i in range(6):
    for j in range(4):
         model = AC(n_clusters = clusters[i], linkage = linkages[j])
         X, y = datasets[i]
         X = StandardScaler().fit_transform(X)
         model.fit(X)
         ax = axes[i][j]
         if i == 0: ax.set_title(linkages[j], fontsize = 20)
         ax.scatter(X[:,0], X[:,1], c = model.labels_, cmap = 'rainbow',s = 10)
         ax.set_xlim(-2.5, 2.5)
         ax.set_ylim(-2.5, 2.5)
         t1 = time.time()
         duration, t0 = t1 - t0, t1
         ax.text(.1, .01, ('%.2fs' % duration).lstrip('0'),transform = ax.transAxes,
         size=10, horizontalalignment='right')
plt.show()


'''
digits = load_digits()
data, labels = digits.data, digits.target
data = PCA(n_components = 2).fit_transform(data)
linkages = ['complete', 'average', 'ward']
fig, axes = plt.subplots(1,3,figsize = (30, 15), subplot_kw = {'xticks':[], 'yticks':[]},
gridspec_kw = {'hspace':.05})
fig.suptitle('Digits Dataset: PCA')
for ax, linkage in zip(axes.flat, linkages):
    model = AC(n_clusters = 10, linkage = linkage)
    y = model.fit_predict(data)
    ax.scatter(data[:,0], data[:,1], c = y, cmap = 'spectral', edgecolor = 'k')
    ax.set_title('linkage:%s' % linkage)
plt.show()
'''
'''
from time import time

import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from sklearn import manifold, datasets

digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target
n_samples, n_features = X.shape

np.random.seed(0)

def nudge_images(X, y):
    # Having a larger dataset shows more clearly the behavior of the
    # methods, but we multiply the size of the dataset only by 2, as the
    # cost of the hierarchical clustering methods are strongly
    # super-linear in n_samples
    shift = lambda x: ndimage.shift(x.reshape((8, 8)),
                                  .3 * np.random.normal(size=2),
                                  mode='constant',
                                  ).ravel()
    X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
    Y = np.concatenate([y, y], axis=0)
    return X, Y


X, y = nudge_images(X, y)


#----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.nipy_spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#----------------------------------------------------------------------
# 2D embedding of the digits dataset
print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done.")

from sklearn.cluster import AgglomerativeClustering

for linkage in ('ward', 'average', 'complete'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    t0 = time()
    clustering.fit(X_red)
    print("%s :\t%.2fs" % (linkage, time() - t0))

    plot_clustering(X_red, clustering.labels_, "%s linkage" % linkage)


plt.show()'''
