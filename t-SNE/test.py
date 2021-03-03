from TSNE import *
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE as tees_nee
from sklearn.datasets import make_circles, make_s_curve
from mpl_toolkits.mplot3d import Axes3D

''' Uncomment each dataset to see How T-SNE works these datasets '''''

#___________________________________________________________________________________
#3 Clusters Guassian Distributioin
'''
cov = np.eye(2)
A =  np.random.multivariate_normal([2,1], cov, size = 50)
B = np.random.multivariate_normal([5,10], cov, size = 50)
C = np.random.multivariate_normal([9, 4], cov, size = 50)

data = np.vstack((A,B,C))
labels = np.array([0] * 50 + [1] * 50 + [2] * 50)'''
#_____________________________________________________________________________________________
'''
A = np.random.normal(scale = 1, size = (100,3))
B = np.array([x for x in np.random.normal(scale = 5, size = (500,3)) if np.linalg.norm(x) > 7])
data = np.vstack((A,B))
labels = np.array([0] * len(A) + [1] * len(B))'''
''' Visualize 3D plot of the dataset
ax = plt.subplot(111, projection = '3d')
ax.scatter(data[:,0], data[:,1], data[:,2],c = labels, cmap = 'spring')
plt.show()'''
#_________________________________________________________________________________________
'''
#Make Circle dataset
data, labels = make_circles(n_samples = 100, noise = .05, factor = .5)'''
#_________________________________________________________________________________
data, labels = make_s_curve(n_samples = 500, random_state = 6)
#_________________________________________________________________________________________
fig, axes = plt.subplots(1,5, figsize = (15, 8), gridspec_kw = {'hspace':.25},
 subplot_kw = {'xticks':[], 'yticks':[]})
fig.suptitle('T-SNE')
perplexities = [2,5,30,50,100]
for ax, perplexity in zip(axes.flat, perplexities):
    #model = TSNE(learning_rate = 200, perplexity = perplexity, n_iter = 1000)
    #X = model.fit(data)
    model = tees_nee(n_components = 2, perplexity = perplexity)
    X = model.fit_transform(data)
    #print(model.Kullback_Leibler_)
    ax.set_xlabel('perplexity: %d' % perplexity)
    #ax.set_title('Kullback_Leibler: %d' % model.Kullback_Leibler_)
    ax.scatter(X[:,0], X[:,1], c = labels, cmap = 'spring')
    ax.axis('tight')

plt.show()
