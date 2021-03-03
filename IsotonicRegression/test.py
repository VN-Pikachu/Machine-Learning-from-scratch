from PAV import *
from sklearn.isotonic import IsotonicRegression
clf = IsotonicRegression()

model = PVA()
n = 20
x = np.arange(n)
#y = np.random.randint(-50, 50, size=(n,)) + 50. * np.log1p(np.arange(n))
y = np.random.randint(1,10,n)
w = np.random.randint(1,100,n)
y_ = clf.fit_transform(x,y, sample_weight = w)
#model.fit(y, np.ones(n))
model.fit(y, w)
y_custom = model.solution_
print(y_)
print('_' * 100)
print(y_custom)
print('_' * 100)
print(y_ == y_custom)
print(np.all(y_==y_custom))

'''
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
plt.subplot(2,1,1)
plt.scatter(x, y, c = 'red', alpha = .5)
plt.plot(x, model.solution_, c = 'green', marker = 'o')
plt.subplot(2,1,2)
plt.scatter(x, y, c = 'red', alpha = .5)
plt.plot(x, y_, c = 'green', marker = 'o')
plt.show()
'''
