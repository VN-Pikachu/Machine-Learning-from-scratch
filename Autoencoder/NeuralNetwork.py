import numpy as np
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoidPrime(z): return sigmoid(z) * (1 - sigmoid(z))
def identity(z): return z
def identityPrime(z): return 1
def relu(z):
    z = np.copy(z)
    z[z < 0] = 0
    return z
def reluPrime(z):
    z = np.copy(z)
    z[z > 0] = 1
    return z
def tanh(z): return np.tanh(z)
def tanhPrime(z): return 1 - tanh(z) ** 2
activations = {'logistic' :sigmoid, 'identity' : identity, 'relu': relu, 'tanh' : tanh}
activationPrimes = {'logistic': sigmoidPrime, 'identity' : identityPrime, 'relu' : reluPrime, 'tanh' : tanhPrime}

class NeuralNetwork:
    def __init__(self, hidden_layer_sizes, learning_rate = 1e-3, batch_size = 'auto', max_iter = 100,
    activation = 'logistic', shuffle = True):
        self.activation = activations[activation]
        self.prime = activationPrimes[activation]
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.N = len(hidden_layer_sizes)
        self.A = [0] * self.N
        self.W = [np.random.randn(hidden_layer_sizes[i], hidden_layer_sizes[i-1]) for i in range(self.N)]
        self.b = [np.random.randn(size) for size in hidden_layer_sizes]
        self.D = [0] * self.N

    def SGD(self, minibatch):
        M = len(minibatch)
        GW = [np.zeros_like(w) for w in self.W]
        Gb = [np.zeros_like(b) for b in self.b]

        for x, y in minibatch:
            #Forward Propagation
            self.A[0] = x
            for i in range(1, self.N):
                self.A[i] = self.activation(self.W[i] @ self.A[i - 1] + self.b[i])
            #backward Propagation
            self.D[-1] = (self.A[-1] - y) * self.prime(self.A[-1])
            for i in range(self.N - 2, 0, -1):
                self.D[i] = (self.W[i + 1].T @ self.D[i + 1]) * self.prime(self.A[i])
            #Backpropagate error
            for i in range(1, self.N):
                GW[i] += np.outer(self.D[i], self.A[i - 1])
                Gb[i] += self.D[i]
        for i in range(1, self.N):
            self.W[i] -= self.learning_rate / M * GW[i]
            self.b[i] -= self.learning_rate / M * Gb[i]
    def fit(self, X, y):
        data = list(zip(X,y))
        N = len(X)
        if self.batch_size == 'auto': self.batch_size = min(200, N)
        for _ in range(self.max_iter):
            #print('Epochs: % d / % d' % (_ + 1, self.max_iter))
            if self.shuffle: np.random.shuffle(data)
            for i in range(0, N, self.batch_size):
                self.SGD(data[i:i + self.batch_size])
            #print(np.mean(self.predict(X) == np.argmax(y, axis = 1)))
        self.coefs_ = self.W[1:]
        self.intercepts_ = self.b[1:]
    def predict_proba(self, X):
        for w, b in zip(self.W[1:], self.b[1:]): X = self.activation(X @ w.T + b)
        return X

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis = 1)
