import matplotlib.pyplot as plt
import numpy as np
class SVM:
    def fit(self, x, y):
        self.m = len(x)
        self.w = np.random.randn(len(x[0]))
        self.SGD(x, y)
    def SGD(self, x, y, LAMBDA = 0.01, eta = 0.1, epochs = 1000):
        for _ in range(1, epochs):
            #cost function
            #print(sum(max(0, 1 - val) for val in x @ self.w * y) + LAMBDA / 2 * self.w @ self.w)
            tmp = x @ self.w * y
            gradient = np.array([0 if tmp[i] >= 1 else -y[i] for i in range(len(tmp))])
            delta = LAMBDA * self.w + x.T @ gradient
            self.w -= eta / self.m * delta
    def predict(self, x):
        return np.sign(x @ self.w)
"""
classifier = SVM()
data = []
for i in range(50):
    x = np.random.randint(low = 0, high = 50)
    sign = 1 if i < 25 else -1
    marker = '+' if i < 25 else '_'
    y = x + sign * np.random.randint(low = 5, high = 20)
    plt.scatter(x,y,marker = marker)
    data.append([x,y, 1])

classifier.fit(np.array(data), np.array([1] * 25 + [-1] * 25))
a, b, c = classifier.w
x = [-5, 50]
y = [(-a * val - c) / b for val in x]
plt.plot(x, y, linewidth = 2, color = 'teal')
print(a, b, c, x, y)
plt.show()"""
