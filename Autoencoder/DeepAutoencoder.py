from NeuralNetworkClone import *

class DAE:
    def __init__(self, latent_layer_size, encode_layer_sizes = [], decode_layer_sizes = [],
    batch_size = 'auto', learning_rate = 1e-3, activation = 'identity', max_iter = 100):
        self.encode_layer_sizes = encode_layer_sizes
        self.latent_layer_size = latent_layer_size
        self.decode_layer_sizes = decode_layer_sizes
        self.batch_size = batch_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
    def fit(self, X):
        N = X.shape[1]
        layer_sizes = [N] + self.encode_layer_sizes + [self.latent_layer_size] + self.decode_layer_sizes + [N]
        model = NeuralNetwork(layer_sizes, activation = self.activation, max_iter = self.max_iter,
        learning_rate = self.learning_rate, batch_size = self.batch_size)
        model.fit(X,X)
        k = len(self.encode_layer_sizes) + 1
        self.encode_coefs_ = model.coefs_[:k]
        self.encode_intercepts_ = model.intercepts_[:k]
        self.decode_coefs_ = model.coefs_[k:]
        self.decode_intercepts_ = model.intercepts_[k:]
        #Take the activation function from  NeuralNetwork for encode and decode
        self.activation = model.activation
    def forward(self, X, W, B):
        for w, b in zip(W, B):
            X = self.activation(X @ w.T + b)
        return X
    def encode(self, X):
        return self.forward(X, self.encode_coefs_, self.encode_intercepts_)
    def decode(self, X):
        return self.forward(X, self.decode_coefs_, self.decode_intercepts_)

from sklearn.datasets import load_iris, load_digits
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
''' Visualize Iris dataset (with just 1 hidden layer)

iris = load_iris()
data, labels = iris.data, iris.target
model = DAE(latent_layer_size = 2,
batch_size = 20, max_iter = 100, activation = 'identity')
model.fit(data)
X = model.encode(data)
plt.scatter(X[:,0], X[:,1], c = labels, cmap = 'spring')
plt.show()'''

'''Visualize digits datasets '''''

digits = load_digits()
from sklearn.preprocessing import StandardScaler

data, labels = digits.data, digits.target
#Increase the size of latent layer will make much more better results
#But when increase latent size we also need to care about decreasing learning learning_rate
#otherwise overflow
model = DAE(latent_layer_size = 16, batch_size = 30, max_iter = 200, activation = 'identity', learning_rate = 1e-5)
model.fit(data)

samples = data[:8]
fig, axes = plt.subplots(8, 2, figsize = (15, 10))
for ax, number, renum in zip(axes, samples, model.decode(model.encode(samples))):
    ax[0].imshow(number.reshape((8,8)), cmap = 'binary')
    ax[1].imshow(renum.reshape((8,8)), cmap = 'binary')
plt.show()
