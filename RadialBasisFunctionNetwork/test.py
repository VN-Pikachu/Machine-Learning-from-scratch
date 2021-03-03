from RBFnets import *
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import pandas as pd
from scipy import sparse
'''' --------------------- Function approximation --------------------------------------'''
'''
NUM_SAMPLES = 100
X = np.random.uniform(0., 1., NUM_SAMPLES)
X = np.sort(X, axis=0)
noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
y = np.sin(2 * np.pi * X)  + noise
X = X.reshape((-1,1))
rbfnet = RBFNetwork(2,1)
rbfnet.fit(X, y.reshape((-1,1)))

y_pred = rbfnet.predict(X)
print(X.shape, y_pred.shape)
plt.plot(X, y, '-o', label='true')
plt.plot(X.flatten(), y_pred.flatten(), '-o', label='RBF-Net')
plt.legend()

plt.tight_layout()
plt.show()
'''


#Works bad for iris_data :0.3% accuracy
def one_hot_coding(labels):
    N, M = len(labels), len(np.unique(labels))
    return sparse.coo_matrix((np.ones(N), (np.arange(N), labels)), shape = (N,M)).toarray()
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
iris = load_iris()
data, labels = iris.data, iris.target
print(np.unique(labels))
data = StandardScaler().fit_transform(data)
labels = one_hot_coding(labels)
x_train, x_test , y_train, y_test = train_test_split(data, labels, train_size = .8, test_size = .2, random_state = 7)
model = RBFNetwork(20, 3)
model.fit(x_train, y_train)
y_pred = np.argmax(model.predict(x_test), axis = 1)
print(np.mean(y_pred == np.argmax(y_test)))
