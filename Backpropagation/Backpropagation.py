import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
#NOTE: instead of converting ndarray to matrix then using matrix muliplication
#      we use '@' operator on 2 ndarrays and still produce the desire result
#      i.e : ndarray_ 1 @ ndarray _ 2 the same as np.matrix(ndarray_1) * np.matrix(ndarray_1)
#NOTE: this does not work for  the case 2 ndarrays are 1 dimesion
#      we MUST convert to matrix to multiply properly
#      i.e : np.array([1, 2]) @ np.array([3,4]) --> return an integer
#            np.matrix([1, 2]) * np.matrix([3, 4, 5]).transpose -> matrix(2 x 3)
class Network:
    def __init__(self, layers, learningRate = 7, iterations = 1000):
        #self.n: the total number of layers
        self.n = len(layers)
        self.iterations = iterations
        self.layers = layers
        self.learningRate = learningRate
        #self.a[l]: a vector of 'activations' of units in layer l
        self.a = [np.random.randn(size) for size in layers]
        self.bias = [np.random.rand(size) for size in layers]
        #self.w[l]: a matrix W: row: units in layer l  ---   col: units iin layer l - 1
        #W(jk): the weight connects unit k in layer l - 1 to unit j in layer l
        self.w = [np.random.randn(layers[i], layers[i - 1]) for i in range(0, self.n)]
        #self.error[l]: a vector of 'error' in layer l
        self.delta = [0] * self.n
    #Generate a vector from an ndarray
    def vector(self, s):
        return np.matrix(s.reshape(-1, 1))
    def forward(self):
        for l in range(1, self.n):
            self.a[l] = self.sigmoid(self.w[l] @ self.a[l - 1] + self.bias[l])
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def sigmoidPrime(self, l):
        return self.a[l] * (1 - self.a[l])
    def backward(self):
        for l in range(self.n - 2, 0, -1):
            self.delta[l] = self.w[l + 1].T @ self.delta[l + 1] * self.sigmoidPrime(l)
    def updateWeight(self):
        for l in range(self.n - 1, 0, -1):
            #Matrix multiplication (explained on top)
            D = self.vector(self.delta[l]) * self.vector(self.a[l - 1]).transpose()
            self.w[l] -= self.learningRate / self.m * D
            self.bias[l] -= self.learningRate / self.m * self.delta[l]
    def fit(self, x_train, y_train):
        #self.m stores the current total number of inputs
        self.m = len(x_train)
        for _ in range(self.iterations):
            for x, y in zip(x_train, y_train):
                self.a[0] = x
                self.forward()
                expected_output = np.zeros(self.layers[-1])
                expected_output[y] = 1
                #Initialize delta for the ouput layer
                self.delta[-1] = (self.a[-1] - expected_output) * self.sigmoidPrime(-1)
                self.backward()
                self.updateWeight()
    def predict(self, x):
        self.a[0] = x
        self.forward()
        return np.argmax(self.a[-1])
    def predict_proba(self, x):
        self.a[0] = x
        self.forward
        return self.a[-1]

classifier = Network([64, 30, 10])
data = load_digits()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, train_size = .8, test_size = .2, random_state = 6)

classifier.fit(x_train, y_train)
#
print('Successfully')
correct = 0
for i in range(len(x_test)):
    correct += classifier.predict(x_test[i]) == y_test[i]
print(correct / len(x_test))
