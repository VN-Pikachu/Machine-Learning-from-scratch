def gen_lin_separable_data():
    # generate training data in the 2-d case
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

def gen_non_lin_separable_data():
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0,0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

from SVM_Kernel import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pl
def plot_margin(X1, X2, clf):
    #wx + b = c
    def f(w, x, b, c):
        return (c - b - w[0] * x) / w[1]

    pl.plot(X1[:, 0], X1[:, 1], 'co', alpha = .8)
    pl.plot(X2[:, 0], X2[:, 1], 'g^',alpha = .8)

    #wx + b = 0
    coord = np.array([-4, 4])
    pl.plot(coord, f(clf.w, coord, clf.b, 0), 'k')
    #wx + b = -1
    pl.plot(coord, f(clf.w, coord, clf.b, 1), 'k--')
    #wx + b = 1
    pl.plot(coord, f(clf.w, coord, clf.b, -1), 'k--')
    #support vectors
    pl.scatter(clf.X[:, 0], clf.X[:, 1], s = 100, c = 'r')
    pl.axis('tight')
    pl.show()
def test_linear():
    x1, y1, x2, y2 = gen_lin_separable_data()
    clf = SVM()
    clf.fit(np.vstack((x1, x2)), np.hstack((y1, y2)))
    plot_margin(x1, x2, clf)

#test_linear()
def plot_contour(X1, X2, clf):
    pl.plot(X1[:, 0], X1[:, 1], 'c^')
    pl.plot(X2[:, 0], X2[:, 1], 'go')
    print(clf.X, type(clf.X))
    pl.scatter(clf.X[:, 0], clf.X[:, 1], s = 100, c = 'r')
    x, y = np.meshgrid(np.linspace(-6, 6, 50), np.linspace(-6, 6, 50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(x), np.ravel(y))])
    Z = clf.project(X).reshape(x.shape)
    pl.contour(x, y, Z, [0.0], colors='k', linewidths=1, origin='lower')
    pl.contour(x, y, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    pl.contour(x,y, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')
    pl.axis('tight')
    pl.show()

def test_non_linear():
    x1, y1, x2, y2 = gen_non_lin_separable_data()
    clf = SVM(polynomial_kernel)
    clf.fit(np.vstack((x1, x2)), np.hstack((y1, y2)))
    plot_contour(x1, x2, clf)

test_non_linear()
