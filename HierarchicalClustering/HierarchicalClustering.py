import numpy as np
from numpy.linalg import norm
from collections import defaultdict
from itertools import combinations
def MIN(X, Y): return min(norm(x - y) for x in X for y in Y)
def MAX(X, Y): return max(norm(x - y) for x in X for y in Y)
def WARD_METHOD(X,Y): return np.mean([norm(x - y) ** 2 for x in X for y in Y])
def MEAN(X,Y): return np.mean([norm(x-y) for x in X for y in Y])
methods = dict(single = MIN, complete = MAX, average = MEAN, ward = WARD_METHOD)
class DSU:
    def __init__(self, n):
        self.set = np.arange(n)
    def find(self, v):
        if self.set[v] == v: return v
        self.set[v] = self.find(self.set[v])
        return self.set[v]
    def union(self, u, v):
        u, v = self.find(u), self.find(v)
        if u == v: return False
        self.set[u] = v
        return True

#Methods: single (min), complete(max), average(group average), ward(ward's method)
class AgglomerativeClustering:
    def __init__(self, n_clusters = 2, linkage = 'ward'):
        self.n_clusters = n_clusters
        self.linkage = methods[linkage]
    def fit(self, X):
        N = len(X)
        dsu = DSU(N)
        for _ in range(N - self.n_clusters):
            set = [dsu.find(u) for u in range(N)]
            clusters = np.unique(set)
            pairs = combinations(clusters, 2)
            print('Epochs:%d' % _)
            u, v = min(pairs, key = lambda p: self.linkage(X[set == p[0]], X[set == p[1]]))
            dsu.union(u,v)
        self.labels_ = np.array([dsu.find(u) for u in range(N)])
