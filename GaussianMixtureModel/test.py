from GaussianMixtureModel import GMM
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.datasets import load_iris, load_digits
'''---------------------- Three Gussian Distributions ---------------------------------------'''
'''
cov = np.eye(2)
A = np.random.multivariate_normal([0,0], cov, 200)
B = np.random.multivariate_normal([6, 9], cov, 200)
P = np.random.multivariate_normal([12, 4], cov, 200)
data = np.vstack((A,B,P))
model = GMM(n_components = 3)
model.fit(data)
labels = model.labels_
plt.scatter(data[:,0], data[:,1], c = labels, cmap = 'rainbow')
plt.show()
'''
from sklearn.decomposition import PCA
digits = load_digits()
iris = load_iris()
#data, labels = iris.data, iris.target
data, labels = digits.data, digits.target
data = PCA(n_components = 2).fit_transform(data)
'''
model = GMM(n_components = 10)
y_pred = model.fit(data)
plt.scatter(data[:,0], data[:,1], c = y_pred, cmap = 'rainbow')
plt.show()
'''
plt.scatter(data[:,0], data[:,1], c = labels, cmap = 'rainbow')
plt.show()
