from SpectralClustering import *
from sklearn.datasets import make_circles, make_moons, make_s_curve
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
X1, _ = make_circles(150, factor = .5, random_state = 6)
X2 = X1 * 4
X = np.vstack((X1, X2))
from sklearn.cluster import SpectralClustering as SC
#clf = SC(n_clusters = 4, gamma = .001)
#clf = SpectralClustering(n_clusters = 4, gamma = .01)
clf = SpectralClustering(n_clusters = 4, affinity = 'nearest_neighbors')
#clf = SpectralClustering(n_clusters = 4, gamma = .1)
clf.fit(X)
plt.subplot(2,1,1)
plt.scatter(X[:,0], X[:,1], c = clf.labels_, cmap = 'rainbow')


def make_hello(N=1000, rseed=42):
    # Make a plot with "HELLO" text; save as PNG
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.text(0.5, 0.4, 'HELLO', va='center', ha='center', weight='bold', size=85)
    fig.savefig('hello.png')
    plt.close(fig)

    # Open this PNG and draw random points from it
    from matplotlib.image import imread
    data = imread('hello.png')[::-1, :, 0].T
    rng = np.random.RandomState(rseed)
    X = rng.rand(4 * N, 2)
    i, j = (X * data.shape).astype(int).T
    mask = (data[i, j] < 1)
    X = X[mask]
    X[:, 0] *= (data.shape[0] / data.shape[1])
    X = X[:N]
    return X[np.argsort(X[:, 0])]

X = make_hello()
plt.subplot(2,1,2)
clf = SpectralClustering(n_clusters = 5, affinity = 'nearest_neighbors')
clf.fit(X)
plt.scatter(X[:,0], X[:,1], c = clf.labels_, cmap = 'rainbow')
plt.show()
