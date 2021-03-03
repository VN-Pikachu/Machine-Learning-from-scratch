from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns;sns.set()
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import locally_linear_embedding as lle
import matplotlib.pyplot as plt
from LLE import *
data, labels = make_swiss_roll(1500, random_state = 100)
model = LLE(2, 20, 0.1)
X = model.fit(data)
print(model.reconstruction_error_)
#X, err = lle(data, n_neighbors = 12, n_components = 2)
fig = plt.figure()
ax = fig.add_subplot(211, projection = '3d')
#print(data)
#print(labels)
ax.scatter(data[:,0], data[:,1], data[:,2], c = labels, s = 20, cmap = 'Spectral')
ax.set_title('Original Data')
ax = fig.add_subplot(212)
ax.scatter(X[:,0], X[:,1], c = labels, s = 20, cmap = 'Spectral')
ax.set_title('Projected Data')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
