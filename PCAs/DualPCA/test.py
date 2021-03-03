from DualPCA import *
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
model = DPCA(n_components = 2)
iris = load_iris()
data,labels = iris.data, iris.target
X = model.fit(data)
plt.scatter(X[:,0], X[:,1], c = labels, cmap = 'rainbow')
plt.show()
