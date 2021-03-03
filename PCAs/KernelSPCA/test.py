from KSPCA import *
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.datasets import make_circles, make_moons, make_swiss_roll, load_digits
from sklearn.decomposition import PCA
''' Make circles
model = KSPCA(n_components = 2, kernel = 'rbf', bandwidth = .02)
pca = PCA(n_components = 2)
data, labels = make_circles(500, noise = .05)
fig, axes = plt.subplots(1,3,subplot_kw = {'xticks':[],'yticks':[]}, figsize = (15,10))
X0 = pca.fit_transform(data)
X1 = model.fit(data, labels.reshape((-1,1)))
axes[0].set_title('Origin')
axes[0].scatter(data[:,0], data[:,1], c = labels, cmap = 'spring')
axes[1].set_title('PCA')
axes[1].scatter(X0[:,0], X0[:,1],c = labels, cmap = 'spring')
axes[2].set_title('KernelSPCA')
axes[2].scatter(X1[:,0], X1[:,1], c = labels, cmap = 'spring')
plt.show()
'''

model = KSPCA(n_components = 2, kernel = 'rbf', bandwidth = .2)
pca = PCA(n_components = 2)

data, labels = make_swiss_roll(500)

fig, axes = plt.subplots(1,2,subplot_kw = {'xticks':[],'yticks':[]}, figsize = (15,10))
X0 = pca.fit_transform(data)
X1 = model.fit(data, labels.reshape((-1,1)))
axes[0].set_title('PCA')
axes[0].scatter(X0[:,0], X0[:,1],c = labels, cmap = 'spring')
axes[1].set_title('KernelSPCA')
axes[1].scatter(X1[:,0], X1[:,1], c = labels, cmap = 'spring')
plt.show()
