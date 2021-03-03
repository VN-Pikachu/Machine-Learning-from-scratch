from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
def GradientDescent(x, y, learningRate = 0.001):
    import numpy as np
    def normalize(x):
        MAX, MIN = np.amax(x, axis = 0), np.amin(x, axis = 0)
        return (x - MIN) / (MAX - MIN)
    m, n = x.shape
    n += 1
    #x = normalize(x)
    x = np.hstack((np.ones((m,1)), x))
    theta = np.random.random(n)
    def cost(x, y, theta):
        s = x @ theta - y
        return s @ s / 2 / m
    iteration = 2000
    X, Y = [], []
    for _ in range(iteration):
        #Optional, Plot the graph of cost function
        #X.append(_)
        #Y.append(cost(x, y, theta))
        error = x @ theta - y
        theta -= learningRate / m * (x.T @ error)
        #optimized when cost changes slowly
        if len(Y) > 1 and Y[-2] - Y[-1] < 1e-3:
            break
    #show plot
    #plt.plot(X, Y)
    #plt.xlabel('iteration')
    #plt.ylabel('cost')
    #plt.title('GradientDescent')
    #plt.show()
    return {'coef_' : theta[1:], 'intercept_' : theta[0]}
