import numpy as np
import matplotlib.pyplot as plt
def LogisticRegression(x, y, learningRate = 1e-5, iteration = 10000):
    from sklearn.model_selection import train_test_split
    m, n = x.shape
    n += 1
    x = np.hstack((np.ones((m, 1)), x))
    x, x_test, y, y_test = train_test_split(x,y, random_state = 4, train_size = .8, test_size = .2)
    theta = np.random.random(n)
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def cost(x, y, theta):
        SUM = sigmoid(x, theta)
        return (y * (np.log(SUM)) + (1 - y) * np.log(1 - SUM)) / -m

    for _ in range(iteration):
        theta -= learningRate / m * (x.T @ (f(x, theta) - y))

    class logistic:
        def __init__(self, coef_, intercept_):
            self.coef_ = coef_
            self.intercept_ = intercept_
        def predict(self, x, threshold = .5):
            return (sigmoid(x @ self.coef_ + self.intercept_) >= threshold).astype('int')
        def predict_proba(self, x):
            return sigmoid(x @ self.coef_ + self.intercept_)
    return logistic(theta[1:], theta[0])
