import numpy as np
from scipy import sparse
#Softmax for a matrix
def softmax(Z):
    Z = np.exp(Z - np.max(Z, axis = 1).reshape((-1,1)))
    return Z / np.sum(Z, axis = 1).reshape((-1,1))
#softmax for a vector
def f(z):
    z = np.exp(z - np.max(z))
    return z / np.sum(z)
#One-hot coding from a vector of labels
def one_hot_coding(labels):
    N, M = len(labels), len(np.unique(labels))
    return sparse.coo_matrix((np.ones(N), (np.arange(N), labels)), shape = (N,M)).toarray()
#Cross-Entropy:
def Cross_Entropy(p, q):
    return - p @ np.log2(q)

class SoftmaxRegression:
    def __init__(self, learning_rate = .5, max_iter = 1000, tol = 1e-4):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol

    def fit(self, X, labels):
        #C = len(np.unique(labels))
        labels = one_hot_coding(labels)
        C = labels.shape[1]
        N, d = X.shape
        #Adding 1 for the intercept
        X = np.hstack((X, np.ones(N).reshape((-1,1))))
        #W is a Matrix with rows as weight vectors:
        #    W[i]: weight vector for class i
        W = np.random.rand(C, d + 1)
        cost = np.inf
        for _ in range(self.max_iter):
            Gradient = np.zeros_like(W)
            for x, Y in zip(X, labels):
                A = f(W @ x)
                Gradient += np.outer(A - Y, x)
            W -= self.learning_rate / N * Gradient
            #Uncomment if want to break when the cost change slowly
            '''
            cur_cost = sum(Cross_Entropy(y, f(W @ x)) for x, y in zip(X, labels))
            print(cur_cost)
            if abs(cur_cost - cost) <= self.tol:break
            cost = cur_cost'''

        self.coef_ = W[:,:-1]
        self.intercept_ = W[:,-1]

    def predict_proba(self, X):
        return softmax(X @ self.coef_.T + self.intercept_)
        
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis = 1)
