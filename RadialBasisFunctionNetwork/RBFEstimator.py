#Radial Basis Function Network for Function approximation, Time Series Prediction
#Take input X(n x d) and return 1D outputs Y : f(x) : X[i] --> Y[i]
import numpy as np
from numpy.linalg import norm, inv
from sklearn.cluster import KMeans
class RBFNetwork:
    def __init__(self, hidden_layer_size):
        self.hidden_layer_size = hidden_layer_size
    def fit(self, X, Y):
        model = KMeans(n_clusters = self.hidden_layer_size)
        model.fit(X)
        self.means_ = model.cluster_centers_
        self.gammas_ = np.zeros(self.hidden_layer_size)
        for k, m in enumerate(self.means_):
            variance = np.mean(norm(X[model.labels_ == k] - m, axis = 1) ** 2)
            self.gammas_[k] = 1 / (2 * variance)
        G = self.transform(X).T
        self.weights_ = inv(G @ G.T) @ G @  Y
    def predict(self, X):
        return self.transform(X) @ self.weights_
    def transform(self, X):
        res = np.zeros((len(X), self.hidden_layer_size))
        for i, x in enumerate(X):
            res[i] = np.exp(-self.gammas_ * norm(x - self.means_, axis = 1) ** 2)
        return res

'''  --------- Function Estimation ------------------------'''
'''
import matplotlib.pyplot as plt;
import seaborn as sns;sns.set();
X = np.sort(np.random.uniform(0., 1., 500))
y = np.sin(X * 2 * np.pi)
model = RBFNetwork(2)
X = X.reshape((-1,1))
model.fit(X, y)
y_pred = model.predict(X)

X = X.flatten()
plt.plot(X, y, '-o', label = 'True')
plt.plot(X, y_pred, '-o', label = 'Predict')
plt.legend()
plt.show()
'''

''' Breast Cancer dataset: 0.94% accuracy ''''
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
breast_cancer = load_breast_cancer()
data, labels = breast_cancer.data, breast_cancer.target
data = StandardScaler().fit_transform(data)
x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = .8, test_size = .2, random_state = 5)
model = RBFNetwork(10)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred = (y_pred > .5).astype('int')
print(np.mean(y_pred == y_test))
