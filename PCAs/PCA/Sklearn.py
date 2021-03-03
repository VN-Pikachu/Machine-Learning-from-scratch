import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.datasets import load_digits
digits = load_digits()
'''
#1.________________________VISUALIZE PCA_________________________
r = np.random.RandomState(1)
X = r.randn(100, 2) @ r.rand(2,2)
def vector(B, A):
    ax = plt.gca()
    ax.annotate('', A, B,
    arrowprops = {'arrowstyle':'->', 'linewidth' :2, 'shrinkA':0, 'shrinkB':0})
model = PCA()
model.fit(X)
plt.scatter(X[:,0], X[:,1], alpha = .8)
for u, w in zip(model.components_, model.explained_variance_):
    vector(model.mean_, model.mean_ + 2.5 * np.sqrt(w) * u)
plt.axis('equal')
plt.show()
'''

'''
#2._____________PCA as demension reduction________ (Hand-written digits)
model = PCA(n_components = 2)
X_new = model.fit_transform(digits.data)
plt.scatter(X_new[:,0], X_new[:,1], c = digits.target, cmap = 'nipy_spectral', s = 10, alpha = .6)
plt.colorbar()
plt.show()'''

'''
#3.____________Chosing the number of components_
model = PCA()
model.fit(digits.data)
plt.plot(range(64), np.cumsum(model.explained_variance_ratio_))
plt.show()
'''

'''
#4_____________Image Compression___________
fig, axes = plt.subplots(3, 6, figsize = (8, 3))
fig.suptitle('Image Compression')
k_dimensions = np.linspace(1, 18, 18).astype('int')
for ax, k in zip(axes.flat, k_dimensions):
    ax.set_title('%d demension(s)' % k, fontsize = 6)
    ax.set(xticks = [], yticks = [])
    model = PCA(n_components = k)
    X = model.fit_transform(digits.data)
    X = model.inverse_transform(X)
    ax.imshow(X[0].reshape((8, 8)), cmap = 'binary', interpolation = 'nearest')
plt.subplots_adjust(hspace = .5)
plt.show()
'''
'''
#5._____________PCA as filtering Noise_________
np.random.seed(42)
data = np.random.normal(digits.data, 2)
def plot_image(data):
    fig, axes = plt.subplots(4, 10, figsize = (10, 4), subplot_kw = {'xticks' : [], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape((8, 8)), interpolation = 'nearest', cmap = 'binary')
    plt.show()
#Noise images, uncomment to see image
#plot_image(data)
#Filter noise
model = PCA(.5)
X = model.fit_transform(data)
filtered = model.inverse_transform(X)
plot_image(filtered)
'''

#6.____________Eigenface______________________________
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person = 60)
model = PCA(n_components = 150)
X = model.fit_transform(faces.data)
'''Plot Principle Components and explained_variance_ratio_
fig, axes = plt.subplots(3,8,figsize =(8,3),
subplot_kw = {'xticks':[], 'yticks':[]},
gridspec_kw = dict(hspace = .1, wspace = .1))
for i, ax in enumerate(axes.flat):
    ax.imshow(model.components_[i].reshape((62,47)), cmap = 'bone')
plt.show()
plt.subplot(1,1,1)
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.show()
'''
images = model.inverse_transform(X)
fig, axes = plt.subplots(2, 10, subplot_kw = {'xticks':[],'yticks':[]}, gridspec_kw=dict(hspace=.1,wspace=.1),
figsize = (10,4))
for i in range(10):
    axes[0, i].imshow(faces.images[i], cmap = 'binary_r')
    axes[1, i].imshow(images[i].reshape((62,47)), cmap = 'binary_r')
plt.show()









'''________________________TESTING CUSTOM PCA________________________________________________________
from PrincipleComponentAnalysis import *
import numpy as np
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt

r = np.random.RandomState(99)
X = r.randn(100, 2) @ r.rand(2,2)
def vector(B, A):
    ax = plt.gca()
    ax.annotate('', A, B, arrowprops = {'arrowstyle' : '->', 'linewidth' : 2, 'shrinkA':0, 'shrinkB':0})

model = PCA(n_components = 1)
X_PCA = model.fit(X)
print(model.components_)
print(model.explained_variance_)
print(model.explained_variance_ratio_)
for u, w in zip(model.components_, model.explained_variance_):
    vector(model.mean_, model.mean_ + 2 * w ** .5 * u)
plt.scatter(X[:,0], X[:,1], alpha = .6)
X_new = model.inverse_transform(X_PCA)
plt.scatter(X_new[:,0], X_new[:,1], color = 'red', alpha = .8)
plt.axis('equal')
plt.show()'''
