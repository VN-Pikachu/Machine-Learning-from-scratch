import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

from KernelPCA import KPCA
from sklearn.datasets import load_iris, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA

fig, axes = plt.subplots(1,2,figsize = (15,10), subplot_kw = {'xticks':[], 'yticks':[]})
fig.suptitle('Kernel Principle Component Analysis')

''' ------------------------------Make Moons ---------------------------------------'''
'''
data, labels = make_moons(n_samples = 200)
axes[0].scatter(data[:,0], data[:,1], c = labels, cmap = 'rainbow')
axes[0].set_title('Origin')
model = KPCA(kernel = 'rbf', bandwidth = 1 / 30 ** .5)
X = model.fit(data)
X = model.transform(data)

#model = KernelPCA(kernel = 'rbf', gamma = , n_components = 2)
#X = model.fit_transform(data)
axes[1].scatter(X[:,0], X[:,1], c = labels, cmap = 'rainbow')
axes[1].set_title('Projection')
axes[1].text(3,3, ' gamma = 1 / sqrt(30) ', fontsize = 30)

plt.tight_layout()
plt.show()
'''
''' ----------------------------Make Circles------------------------------------------'''
'''
data, labels = make_circles(n_samples = 500, noise = .05, factor = .3, random_state = 6)
axes[0].scatter(data[:,0], data[:,1], c = labels, cmap = 'rainbow')
axes[0].set_title('Origin')
model = KPCA(kernel = 'rbf', bandwidth = 1 / 20 ** .5)
X = model.fit(data)
axes[1].scatter(X[:,0], X[:,1], c = labels, cmap = 'rainbow')
axes[1].set_title('Projection')
plt.tight_layout()
plt.show()
'''
''' ------------------------ Iris ---------------------------'''
'''
iris = load_iris()
data, labels = iris.data, iris.target

print('What is going on here')
dummpy = [[1,2,3,4]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = .8, test_size = .2,random_state = 6)
model = KPCA(n_components = 2, kernel = 'linear')
X = model.fit(data)

plt.scatter(X[:,0], X[:,1], c = labels, cmap = 'rainbow')
X = model.transform(data)
plt.scatter(X[:,0], X[:,1], marker = '^')
plt.show()
'''

''''-------------------------- scikitlearn -------------------------------------------------------'''

from keras.datasets.mnist import load_data
(x_train, y_train), (x_test, y_test) = load_data
