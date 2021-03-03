from Meanshift import *
mean = [[5, 8], [10, 7], [7, 14]]
cov = np.identity(2)
N = [100, 150, 60]
X1 = np.random.multivariate_normal(mean[0], cov, N[0])
X2 = np.random.multivariate_normal(mean[1], cov, N[1])
X3 = np.random.multivariate_normal(mean[2], cov, N[2])
y = np.array([0] * N[0] + [1] * N[1] + [2] * N[2])
X = np.vstack((X1, X2, X3))

'''Using Customized Meanshift
mc = ['y^', 'co', 'gs']
for p, label in zip(X, y):
    plt.plot(p[0], p[1], mc[label], alpha = .2)

clf = Meanshift(h = .75, max_iter = 10)
clf.fit(X)
centroids = clf.cluster_centers_
print(centroids)
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', color = 'red')
plt.show()'''

#Using sklearn
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
centers = [[1, -1], [1,1], [-1, -1]]
X = np.array(make_blobs(n_samples = 5000, centers = centers, cluster_std = .6)[0])
#print(X)
#print(X.shape)

bandwidth = estimate_bandwidth(X, quantile = .2, n_samples = 100)
clf = MeanShift(bandwidth = bandwidth, bin_seeding = True)
clf.fit(X)
centroids, labels = clf.cluster_centers_, clf.labels_
plt.scatter(centroids[:, 0], centroids[:, 1], marker = '*', color = 'red')
colors = ['cyan', 'orange', 'gold']
for x, i in zip(X, labels):
    plt.scatter(x[0], x[1], color = colors[i], marker = 'o')
plt.show()
