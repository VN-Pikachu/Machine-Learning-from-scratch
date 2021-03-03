from CMDS import *
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import pandas as pd
from scipy.spatial import distance_matrix
from mpl_toolkits.mplot3d import Axes3D

''' ------Plot similarity between different class of animals------
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
data = pd.read_csv('./datasets/zoo.csv')
labels = data.iloc[:,-1]
names = [0,'Mammal', 'Bird', 'Reptile', 'Fish', 'Amphibian', 'Bug', 'Invertebrate']
colors = [0,'red', 'green', 'blue', 'yellow', 'pink', 'purple', 'brown']
V = data.iloc[:,1:-1]
D = distance_matrix(V, V)
model = CMDS(3)
X = model.fit(D)
for k in range(1, 8):
    m = X[labels == k]
    ax.scatter(m[:,0], m[:,1], m[:,2], c = colors[k], label = names[k])
plt.legend(loc = 3)
plt.show()'''


''' -------Construct tetrahedron -----------
from itertools import combinations
D = np.diag([-1] * 4)
model = CMDS(3)
X = model.fit(D)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
for u, v in combinations(X, 2):
    p = np.vstack((u,v))
    ax.plot(p[:,0], p[:,1], p[:,2], marker = 'o', markersize = 5, linewidth = 2)
plt.show()'''

''' ---- Character Confusion ----- '''
names = 'CDGHMNQW'
colors = ['red', 'green', 'blue', 'purple', 'teal', 'yellow', 'brown', 'pink']
D = [[ 0 ,16 , 9 ,19, 19, 19, 12, 20],
 [16,  0 ,19 ,17, 18 ,17  ,1, 16],
 [ 9, 19 , 0 ,18 ,19, 20, 12 ,19],
 [19, 17 ,18 , 0 , 2 , 3 ,20, 16],
 [19 ,18, 19 , 2 , 0 , 5 ,19,  3],
 [19 ,17, 20 , 3 , 5 , 0,13,  8],
 [12 , 1, 12 ,20, 19, 13 , 0, 17],
 [20, 16, 19 ,16 , 3 , 8 ,17,  0]]
model = CMDS(2)
#model = CMDS(3)
fig = plt.figure()
#ax = fig.add_subplot(111, projection = '3d')
ax = fig.add_subplot(111)
X = model.fit(np.array(D))

for color, n, x in zip(colors, names, X):
    ax.scatter(*x, s = 25, c = color)
    ax.text(*x,n)
plt.show()
