#General version of Radial Basis Function Network
#Takes input X(n x d) (d is the number of dimensions of samples drawed from X)
#Return ouput Y(n x q) (q is the number of units in the output layer)
import numpy as np
from numpy.linalg import norm, inv
from sklearn.cluster import KMeans
class RBFNetwork:
    def __init__(self, hidden_layer_size, output_layer_size):
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
    def fit(self, X, Y):
        N = len(X)
        model = KMeans(n_clusters = self.hidden_layer_size)
        model.fit(X)
        #Locate the center of each cluster
        self.means_ = model.cluster_centers_
        #Find the opproriate value of gamma for each cluster
        self.gammas_ = np.zeros(self.hidden_layer_size)
        for k, mean in enumerate(self.means_):
            variance = np.mean(norm(X[model.labels_ == k] - mean, axis = 1) ** 2)
            self.gammas_[k] = 1 / (2 * variance)

        theta = self.transform(X).T
        W = inv(theta @ theta.T) @ theta @ Y
        #W[i][j]: the weight of the edge connecting unit i in the first layer to unit j
        #         in the second layer (Each column is information about units in the second layer)
        self.weights_ = W
    def predict(self, X):
        return self.transform(X) @ self.weights_
    #Map input data to "RBF activation" layer
    def transform(self, X):
        N = len(X)
        theta = np.zeros((N, self.hidden_layer_size))
        #Row[i] of theta is the 'RBF activation' layer of the X[i] input
        for i, x in enumerate(X):
            theta[i] = np.exp(- self.gammas_ * norm(x - self.means_, axis = 1) ** 2)
        return theta
