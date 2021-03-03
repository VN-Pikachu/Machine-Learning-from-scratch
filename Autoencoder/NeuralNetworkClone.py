import numpy as np
from scipy import sparse
def sigmoid(z): return 1 / (1 + np.exp(-z))

def relu(z):
    z = np.copy(z)
    z[z < 0] = 0.
    return z

def identity(z): return z
def sigmoidPrime(z): return sigmoid(z) * (1 - sigmoid(z))

def reluPrime(z):
    z = np.copy(z)
    z[z > 0] = 1.
    return z
def tanh(z): return np.tanh(z)
def tanhPrime(z): return 1 - tanh(z) ** 2
def identityPrime(z): return 1
activations = {'logistic': sigmoid, 'identity' : identity, 'relu': relu, 'tanh': tanh}
activationPrimes = {'logistic':sigmoidPrime, 'identity':identityPrime, 'relu':reluPrime, 'tanh' : tanhPrime}
def one_hot_coding(labels):
    N, M = len(labels), len(np.unique(labels))
    return sparse.coo_matrix((np.ones(N), (np.arange(N), labels)), shape = (N,M)).toarray()

class NeuralNetwork:
    def __init__(self,  hidden_layer_sizes, learning_rate = 1e-3,
    batch_size = 'auto', max_iter = 200, shuffle = True, activation = 'ReLU'):
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.activation = activations[activation]
        self.prime = activationPrimes[activation]
        self.N = len(hidden_layer_sizes)
        self.W = [np.random.randn(hidden_layer_sizes[i], hidden_layer_sizes[i - 1]) for i in range(self.N)]
        self.b = [np.random.randn(N) for N in hidden_layer_sizes]
        self.D = [0]  * len(hidden_layer_sizes)
        self.A = [0] * len(hidden_layer_sizes)
    def forward(self, x):
        self.A[0] = x
        for i in range(1,self.N):
            self.A[i] = self.activation(self.W[i] @ self.A[i - 1] + self.b[i])
        return self.A[-1]
    def backward(self, y):
        self.D[-1] = (self.A[-1] - y) * self.prime(self.A[-1])
        for i in range(self.N - 2, 0, -1):
            self.D[i] = self.W[i + 1].T @ self.D[i + 1] * self.prime(self.A[i])

    def SGD(self, minibatch):
        self.M = len(minibatch)
        #Weights Gradient
        WG = [np.zeros_like(w) for w in self.W]
        #Biases Gradient
        bG = [np.zeros_like(b) for b in self.b]

        for x, y in minibatch:
            self.forward(x)
            self.backward(y)
            for i in range(1, self.N):
                WG[i] += np.outer(self.D[i], self.A[i - 1])
                bG[i] += self.D[i]

        for i in range(1, self.N):
            self.W[i] -= self.learning_rate / self.M * WG[i]
            self.b[i] -= self.learning_rate / self.M * bG[i]

    def fit(self, X, y):
        data = list(zip(X, y))
        if self.batch_size == 'auto': self.batch_size = min(200, len(X))
        for _ in range(self.max_iter):
            print('Epoch: %d / %d' % (_ + 1, self.max_iter))
            if self.shuffle: np.random.shuffle(data)
            for i in range(0, len(X), self.batch_size):
                self.SGD(data[i:i + self.batch_size])
            print(sum(np.linalg.norm(self.forward(x) - y) ** 2 for x, y in data) / len(X))
        self.coefs_ = self.W[1:]
        self.intercepts_ = self.b[1:]

    def predict_proba(self, X):
        return np.array([self.forward(x) for x in X])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis = 1)
