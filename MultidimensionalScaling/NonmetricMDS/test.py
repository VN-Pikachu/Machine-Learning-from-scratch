from NMDS import *
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.datasets import load_iris
from sklearn.manifold import MDS
iris = load_iris()
'''
model = NMDS(max_iter = 30, alpha = .4)
X, stress = model.fit(iris.data)'''

model = MDS(n_components = 2, metric = False, eps = 1e-12)
X = model.fit_transform(iris.data)
plt.subplot(2,1,1)
plt.scatter(X[:,0], X[:,1], c = iris.target, cmap = 'spring')
plt.show()
