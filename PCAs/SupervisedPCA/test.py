from SPCA import *
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.datasets import load_iris, make_moons, make_circles
from sklearn.decomposition import PCA
''' Iris dataset
iris = load_iris()
data, labels = iris.data, iris.target
'''
cov = [[1,3], [3,1]]
a = np.random.multivariate_normal([0,0], cov, 200)
b = np.random.multivariate_normal([10,3], cov, 200)
data = np.vstack((a,b))
labels = np.array([0] * 200 + [1] * 200)

model = SPCA(n_components = 2, kernel = 'rbf', bandwidth = .2)
pca = PCA(n_components = 2)

fig, axes = plt.subplots(1,3,subplot_kw = {'xticks':[],'yticks':[]}, figsize = (15,10))
X0 = pca.fit_transform(data)
X1 = model.fit(data, labels.reshape((-1,1)))
axes[0].set_title('Origin')
axes[0].scatter(data[:,0], data[:,1], c = labels, cmap = 'spring')
axes[1].set_title('PCA')
axes[1].scatter(X0[:,0], X0[:,1],c = labels, cmap = 'spring')
axes[2].set_title('SPCA')
axes[2].scatter(X1[:,0], X1[:,1], c = labels, cmap = 'spring')
plt.show()
