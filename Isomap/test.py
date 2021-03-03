from Isomap import *
#Visualize 2D Swiss-roll dataset with Isomap
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap as I
'''   #Visualize 2D Swiss-roll dataset with Isomap
fig, axes = plt.subplots(1,4, figsize = (15, 8), subplot_kw = {'xticks':[], 'yticks':[]})
data, labels = make_swiss_roll(100, random_state = 101)
for i, ax in zip([12,15,18,21], axes):
    k = i + 1
    model = Isomap(k, 2)
    #model = I(n_components = 2, n_neighbors = k)
    #X = model.fit_transform(data)
    X = model.fit(data)
    ax.scatter(X[:,0], X[:,1], c = labels, cmap = 'Spectral')
    ax.set_xlabel('k = %d' % k)
plt.show()
'''
'''
#Visualize digits datasets from sklearn
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
import numpy as np
digits = load_digits()
print(len(digits.data))
model = Isomap(n_components = 2, n_neighbors = 20)
colors = ['red', 'green', 'blue', 'violet', 'purple', 'teal', 'lime', 'pink', 'brown', 'yellow']
X = model.fit_transform(digits.data)
for number in np.unique(digits.target):
    subset = X[digits.target == number]
    plt.scatter(subset[:,0], subset[:,1], label = number, color = colors[number])
plt.legend()
plt.show()
'''

''''
#Visualize the iris data
from sklearn.datasets import load_iris
from sklearn.manifold import Isomap as Iso
iris = load_iris()
colors = ['red', 'green', 'blue', 'yellow']
#model = Isomap(n_components = 2, n_neighbors = 20)
#X = model.fit(iris.data)
model = Iso(n_components = 2, n_neighbors = 15)
X = model.fit_transform(iris.data)
for type in np.unique(iris.target):
    subset = X[iris.target == type]
    plt.scatter(subset[:,0], subset[:,1], color = colors[type], label = iris.target_names[type])
plt.legend()
plt.show()
'''

#Swiss-roll with Sklearn
from sklearn.manifold import Isomap as Iso
from sklearn.datasets import make_swiss_roll
data, labels = make_swiss_roll(1000)
fig, axes = plt.subplots(2,4,figsize = (10, 8), subplot_kw = {'xticks':[], 'yticks':[]})
fig.suptitle('Isomap')
for n_neighbors, ax in zip(range(6,35, 4), axes.flat):
    model = Iso(n_components = 2, n_neighbors = n_neighbors)
    X = model.fit_transform(data)
    ax.scatter(X[:,0],X[:,1], c = labels, cmap = 'Spectral')
    ax.set_xlabel('KNN: %s' % n_neighbors)

plt.show()
