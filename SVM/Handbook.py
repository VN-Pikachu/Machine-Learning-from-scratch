import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
#Linear kernel
'''
bankdata = pd.read_csv('./datasets/bill_authentication.csv')
y = bankdata['Class']
X = bankdata.drop('Class', axis = 1)
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = .8, test_size = .2, random_state = 6)
model = svm.SVC(kernel = 'linear')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
sns.heatmap(confusion, square = True, annot = True, cmap = 'Blues')
plt.xlabel('Predict Values')
plt.ylabel('Desire Values')
plt.show()
print(report)'''

#SVM Kernel
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = .8, test_size = .2, random_state = 0)
model = svm.SVC(kernel='poly', degree  = 3)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
confusion = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion, square = True, annot = True, cmap = 'Oranges',
cbar = True, xticklabels = False, yticklabels = False)
plt.show()
print(classification_report(y_test, y_pred))
