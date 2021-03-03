from LogisticRegression import LogisticRegression
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
data, labels = load_breast_cancer(True)
x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state = 6, train_size = .8, test_size =.2)
classifier = LR()
classifier.fit(x_train, y_train)
z = classifier.predict(data)
c = LogisticRegression(data, labels, .0015)
t = c.predict(data)
print(classifier.coef_)
print(c.coef_)
