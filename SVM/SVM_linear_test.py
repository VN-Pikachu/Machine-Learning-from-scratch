from SVM_linear import *
classifier = SVM()
data = []
for i in range(50):
    x = np.random.randint(low = 0, high = 50)
    sign = 1 if i < 25 else -1
    marker = '+' if i < 25 else '_'
    y = x + sign * np.random.randint(low = 5, high = 20)
    plt.scatter(x,y,marker = marker)
    data.append([x,y, 1])
data = np.array(data)
labels = np.array([1] * 25 + [-1] * 25)
classifier.fit(data, labels)
a, b, c = classifier.w


def plot(a, b, c, color = 'teal', label = 'Unknown'):
   x = [-5, 50]
   y = [(-a * val - c) / b for val in x]
   plt.plot(x, y, linewidth = 2, color = color, label = label)


plot(*classifier.w, 'red', 'Gradient Descent')
from sklearn.datasets import load_breast_cancer
from sklearn import svm

s = svm.LinearSVC()
s.fit(data[:,:2], labels)

plot(*s.coef_.flatten(), s.intercept_, 'blue', 'Sklearn')

from SVM_Kernel import SVM
classifier = SVM()
classifier.fit(data[:,:-1], labels)
plot(*classifier.w, classifier.b, 'gold', 'QP')
print(classifier.w, classifier.b)

plt.legend()
plt.show()
