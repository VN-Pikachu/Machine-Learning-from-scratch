import numpy as np
def GradientDescent(x, y, learningRate = 0.001):

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
    iteration = 10000
    X, Y = [], []
    for _ in range(iteration):
        #Optional, Plot the graph of cost function
        #X.append(_)
        #Y.append(cost(x, y, theta))
        error = x @ theta - y
        theta -= learningRate / m * (x.T @ error)
        #optimized when cost changes slowly
        if len(Y) > 1 and Y[-2] - Y[-1] < 1e-5:
            break
    #show plot
    #plt.plot(X, Y)
    #plt.xlabel('iteration')
    #plt.ylabel('cost')
    #plt.title('GradientDescent')
    #plt.show()
    return {'coef_' : theta[1:], 'intercept_' : theta[0]}


X = np.random.randint(low = 0, high = 200, size = 300)
Y = [-5 * val + 50 + np.random.uniform(-30, 30) for val in X]
import matplotlib.pyplot as plt
classifier = GradientDescent(np.array([[val] for val in X]), Y, 1e-6)
a, b = classifier['coef_'], classifier['intercept_']
Z = a * X + b
plt.scatter(X, Y, color='blue', marker = 'o')
plt.plot(X, Z, linewidth = 3, color = 'red')
plt.legend(['line'])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Y-label')
plt.show()
