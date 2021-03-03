from scipy.spatial.distance import cdist
from sklearn.mixture import GMM
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt;
import numpy as np
import seaborn as sns;sns.set();
data, labels = make_blobs(400, cluster_std = .6, random_state = 0, centers = 4)
#Skew the data
data = data @ [[0.2,0.6], [0.5,.4]]
from matplotlib.patches import Ellipse
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=25, cmap='viridis', zorder=2, edgecolor = 'k')
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
model = GMM(n_components = 4, covariance_type = 'full')
plot_gmm(model, data)
plt.title('Guassian Mixture Model\n covariance_type: full')
plt.show()
