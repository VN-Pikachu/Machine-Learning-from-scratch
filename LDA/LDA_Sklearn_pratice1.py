import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.model_selection import train_test_split

data = np.loadtxt('./datasets/wine.data', delimiter = ',')
X = data[:, 1:]
y = data[:,0]
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = .8, test_size = .2, random_state = 6)
model = LDA(n_components = 2)
model.fit(x_train, y_train)
X_new = model.transform(X)
plt.scatter(X_new[:,0], X_new[:,1], c = y, cmap = 'rainbow')
plt.show()
print(model.score(x_test, y_test))
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), square = True, annot = True, cmap = 'Greens')
plt.show()
