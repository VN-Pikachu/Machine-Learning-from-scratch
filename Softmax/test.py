from SoftmaxRegression import *
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
iris = load_iris()
digits = load_digits()
wine = load_wine()
cancer = load_breast_cancer()
data, labels = iris.data, iris.target
data = StandardScaler().fit_transform(data)
x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = .8, test_size = .2, random_state = 6)
model = SoftmaxRegression(learning_rate = .5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(np.mean(y_pred == y_test))
