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
def tanhPrime(z): return 1 - np.tanh(z)**2

activations = {'logistic' :sigmoid, 'identity' : identity, 'relu': relu, 'tanh':tanh}
activationPrimes = {'logistic': sigmoidPrime, 'identity' : identityPrime, 'relu' : reluPrime, 'tanh':tanhPrime}

class SAE:
    def __init__(self, hidden_layer_sizes, learning_rate = 1e-3, batch_size = 'auto', max_iter = 100,
    activation = 'logistic', shuffle = True, beta = 0., sparsity = .05):
        self.beta = beta
        self.sparsity = sparsity
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
        #Expectation of activation for each units in the Neural Network:
        #    E[l][i]: Expectation of activation of the unit i on layer l over the minibatch
        E, X  = [0], np.array([x for x, y in minibatch])

        #Forward Propagation to calculate Expectation for each unit

        for i in range(1, self.N):
            X = self.activation(X @ self.W[i].T + self.b[i])
            E.append(np.mean(X, axis = 0))

        for x, y in minibatch:
            #Forward Propagation
            self.A[0] = x
            for i in range(1, self.N):
                self.A[i] = self.activation(self.W[i] @ self.A[i - 1] + self.b[i])
            #backward Propagation

            self.D[-1] = (self.A[-1] - y) * self.prime(self.A[-1])
            #print(self.D[-1])
            for i in range(self.N - 2, 0, -1):
                p, p_ = self.sparsity, E[i]
                sparse_error = self.beta * (-p / p_ +  (1-p) / (1-p_))
                self.D[i] = (self.W[i + 1].T @ self.D[i + 1] + sparse_error) * self.prime(self.A[i])

            #Backpropagate error
            for i in range(1, self.N):
                GW[i] += np.outer(self.D[i], self.A[i - 1])
                Gb[i] += self.D[i]
        #Update weights and biases
        for i in range(1, self.N):
            self.W[i] -= self.learning_rate / M * GW[i]
            self.b[i] -= self.learning_rate / M * Gb[i]

    def fit(self, X, y):
        data = list(zip(X,y))
        N = len(X)
        if self.batch_size == 'auto': self.batch_size = min(200, N)
        for _ in range(self.max_iter):
            print('Epoch: %d / %d' % (_ + 1, self.max_iter))
            if self.shuffle: np.random.shuffle(data)
            for i in range(0, len(X), self.batch_size):
                self.SGD(data[i:i + self.batch_size])
            print(sum(np.linalg.norm(self.forward(x) - y) ** 2 for x, y in data) / len(X))

        self.encode_coefs_ = self.W[1]
        self.encode_intercepts_ = self.b[1]
        self.decode_coefs_ = self.W[2]
        self.decode_intercepts_ = self.b[2]

    def encode(self, X):
        return X @ self.encode_coefs_.T + self.encode_intercepts_
    def decode(self, X):
        return X @ self.decode_coefs_.T + self.decode_intercepts_
    def forward(self, x):
        self.A[0] = x
        for i in range(1,self.N):
            self.A[i] = self.activation(self.W[i] @ self.A[i - 1] + self.b[i])
        return self.A[-1]
