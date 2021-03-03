import numpy as np
from scipy import sparse
def one_hot_coding(labels):
    N, M = len(labels), len(np.unique(labels))
    return sparse.coo_matrix((np.ones(N), (np.arange(N), labels)), shape = (N,M)).toarray()
def softmax(Z):
    Z  = np.exp(Z - np.max(Z, axis = 1).reshape((-1,1)))
    return Z / np.sum(Z, axis = 1).reshape((-1,1))
def f(z):
    z = np.exp(z - np.max(z))
    return z / np.sum(z)
class SoftmaxRegression:
    def __init__(self, learning_rate = 1e-3, batch_size = 'auto', shuffle = True, max_iter = 100):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = True
        self.max_iter = max_iter
    def SGD(self, data):
        G = np.zeros_like(self.W)
        for x, y in data:
            A = f(self.W @ x)
            G += np.outer(A - y, x)
        self.W -= self.learning_rate / len(data) * G
    def fit(self, X, y):

        N, d = X.shape
        if  self.batch_size == 'auto': self.batch_size = min(200,N)
        C = len(np.unique(y))
        X = np.hstack((X, np.ones(N).reshape((-1,1))))
        self.W = np.random.rand(C, d + 1)
        data = list(zip(X, one_hot_coding(y)))
        for _ in range(self.max_iter):
            if self.shuffle: np.random.shuffle(data)
            for i in range(0, N,self.batch_size):
                self.SGD(data[i:i + self.batch_size])
        self.coefs_ = self.W[:,:-1]
        self.intercepts_ = self.W[:,-1]
    def predict_proba(self, X):
        return softmax(X @ self.coefs_.T + self.intercepts_)
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis = 1)

from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
digits = load_digits()
iris = load_iris()
#data, labels = digits.data, digits.target
data, labels = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = .8, test_size = .2, random_state = 8)
#model = SoftmaxRegression(batch_size = 30)
model = SoftmaxRegression(max_iter = 1000, learning_rate = .5, shuffle = False)
model.fit(x_train, y_train)
print(np.mean(y_test == model.predict(x_test)))
