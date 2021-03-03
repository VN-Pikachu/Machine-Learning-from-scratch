from scipy.spatial.distance import cdist
from sklearn.mixture import GMM
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt;
import numpy as np
import seaborn as sns;sns.set();
data, labels = make_blobs(400, cluster_std = .6, random_state = 0, centers = 4)
#Skew the data
data = data @ [[0.2,0.6], [0.5,.4]]
model = GMM(n_components = 4, covariance_type = 'full')
y = model.fit_predict(data)
plt.scatter(data[:,0], data[:,1], c = y, cmap = 'rainbow', edgecolor = 'k', s = 25)
plt.title('Gaussian Mixture Model \nCovariance_type : full')
plt.show()
