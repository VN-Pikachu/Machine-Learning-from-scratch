from PrincipleComponentAnalysis import *
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
iris = load_iris()
clf1 = PCA(.9)
clf2 = sklearnPCA(n_components = 2)
a = clf1.fit(iris.data)
data = StandardScaler().fit_transform(iris.data)
clf2.fit(data)
b = clf2.transform(data)
print(clf1.components_)
print(clf2.components_)
print(clf1.explained_variance_)
print(clf2.explained_variance_)
for i in range(10):
    print(a[i], '------', b[i])

def plot(X, y):
    colors = ['red', 'orange', 'blue']
    markers = ['^', 'd', 'o']
    for u, v in zip(X, y):
        plt.scatter(u[0], u[1], color = colors[v], marker = markers[v], alpha = .8)
plt.subplot(2,1,1)
plot(a, iris.target)
plt.subplot(2,1,2)
plot(b, iris.target)
plt.show()
