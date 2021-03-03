from NeuralNetworkClone import *
from sklearn.preprocessing import StandardScaler
class UAU:
    def __init__(self, hidden_layer_size = 2, activation = 'relu', batch_size = 'auto',
    max_iter = 100, learning_rate = 1e-3):
        self.hidden_layer_size =  hidden_layer_size
        self.activation = activation
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
    def fit(self, X):
        N, M = X.shape
        model = NeuralNetwork([M, self.hidden_layer_size, M], activation = self.activation,
        batch_size = self.batch_size, max_iter = self.max_iter)
        model.fit(X, X)
        self.coefs_ = model.coefs_
        self.intercepts_ = model.intercepts_
    def encode(self, X):
        return X @ self.coefs_[0].T + self.intercepts_[0]
    def decode(self, X):
        return X @ self.coefs_[1].T + self.intercepts_[1]

from sklearn.datasets import load_iris, load_digits, make_s_curve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

''' ----------------PCA vs Autoencoder ------------------------------------
#When the activations are linear -> Autoencoder is PCA
iris = load_iris()
data, label = iris.data, iris.target
digits = load_digits()

#data, label = digits.data, digits.target
#data, label = make_s_curve(n_samples = 1500)
data = StandardScaler().fit_transform(data)
colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown', 'yellow', 'teal', 'lime']
model = UAU(batch_size = 25, max_iter = 500, activation = 'identity')
model.fit(data)

X1 = model.encode(data)

X2 = PCA(n_components = 2).fit_transform(data)
def plot(ax, X, title):
    for y in np.unique(label):
        p = X[label == y]
        ax.scatter(p[:,0], p[:,1], c = colors[int(y)], label = y)
        ax.set_xlabel(title)
fig, axes = plt.subplots(2,1, figsize = (15,10))

plot(axes[0], X1, 'Undercomplete Autoencoder')
plot(axes[1], X2, 'PCA')
plt.legend()
plt.show()
'------------------------------------------------------------------------------'''
''' S-curve dataset ReLU-----------------------------------------------------------------------
data, labels = make_s_curve(1500)
model = UAU(batch_size = 35, activation = 'relu')
model.fit(data)
X1 = model.encode(data)
X2 = PCA(n_components = 2).fit_transform(data)
plt.subplot(2,1,1)
plt.scatter(X1[:,0], X1[:,1], c = labels, cmap = 'Spectral')
plt.title('Autoencoder: ReLU activation')
plt.subplot(2,1,2)
plt.scatter(X2[:,0], X2[:,1], c = labels, cmap = 'Spectral')
plt.title('PCA')
plt.show()
------------------------------------------------------------------------------------------'''
from sklearn.datasets import load_digits

digits = load_digits()
data, labels = digits.data, digits.target
data /= 16
model = UAU(hidden_layer_size = 16, batch_size = 30, max_iter = 50, activation = 'identity', learning_rate = 3)
model.fit(data)
fig, axes = plt.subplots(8,8, figsize = (15,10), subplot_kw = {'xticks': [], 'yticks':[]})
X1 = data[:32]
X2 = model.decode(model.encode(X1))
axes = axes.flat
def image(z): return z.reshape((8,8))
for a, b in zip(X1, X2):
    ax1 = next(axes)
    ax2 = next(axes)
    ax1.imshow(image(a) , cmap = 'binary')
    ax2.imshow(image(b) , cmap = 'binary')

plt.tight_layout()
plt.grid()
plt.show()
