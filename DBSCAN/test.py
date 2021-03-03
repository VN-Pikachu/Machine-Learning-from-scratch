from DBSCAN import *
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt;
from sklearn.cluster import DBSCAN as sk_DBSCAN

import seaborn as sns;sns.set()
N_SAMPLES = 100
#data, labels = make_blobs(500, centers = 3, cluster_std = [1.5, 2., .5], random_state = 101)
data, labels = make_moons(N_SAMPLES, noise = .05, random_state = 11)
data = StandardScaler().fit_transform(data)
fig, axes = plt.subplots(3,3,figsize = (15,10), subplot_kw = {'xticks':[], 'yticks':[]}, gridspec_kw = {'wspace':.09})
fig.suptitle('Number of samples: %d ' % N_SAMPLES)
epsilons = np.linspace(.1, .8, 9)
n = 10
for eps, ax in zip(epsilons, axes.flat):
    model = DBSCAN(eps, n)
    #model = sk_DBSCAN(eps = eps, min_samples = n)
    model.fit(data)
    noise = data[model.labels_ == -1]
    #Plot noise with black color
    ax.scatter(data[:,0], data[:,1], color = 'black')
    d = data[model.labels_ != -1]
    labels = model.labels_[model.labels_ != -1]
    ax.scatter(d[:,0], d[:,1], c = labels, cmap = 'rainbow')
    ax.set_title('eps = %f, min_samples = %f' % (eps, n), fontsize = 9)
plt.show()
