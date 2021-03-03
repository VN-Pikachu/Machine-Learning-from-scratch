from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
df = pd.read_excel('./datasets/titanic.xls')
df['cabin'].fillna('', inplace = True)
df.fillna(-1, inplace = True)
for column in ['sex', 'cabin']:
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df[column])
    df[column] = encoder.transform(df[column])
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'cabin']
X, y = df[features], df['survived']
print(X.dtypes)
print(X.head(5))
X = preprocessing.scale(X)
clf = KMeans(n_clusters = 2)
clf.fit(X)
clf1 = SVC()
clf1.fit(X, y)
clf2 = LogisticRegression()
clf2.fit(X, y)
print(clf1.score(X, y))
print(clf2.score(X, y))
print(np.mean(clf.labels_ == y))
