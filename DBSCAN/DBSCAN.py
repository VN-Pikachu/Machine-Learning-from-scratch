import numpy as np
from sklearn.neighbors import radius_neighbors_graph

class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
    def fit(self, X):
        #preprocessing data
        N = len(X)
        D = radius_neighbors_graph(X, self.eps).toarray()
        G = np.zeros((N,N))
        #Indices of core points
        self.core_sample_indices_ = np.where(np.sum(D, axis = 1) >= self.min_samples)[0]
        #Go through each core point, set an edge to its epsilon neighborhood
        for core in self.core_sample_indices_:
            neighbors = np.nonzero(D[core])[0]
            for v in neighbors:
                G[core, v] = 1
                G[v, core] = 1

        #Search part
        global visited, labels
        visited = np.zeros(N)
        labels = np.zeros(N) - 1
        k = 0
        def dfs(u, k):
            global visited, labels
            visited[u] = 1
            labels[u] = k

            if u not in self.core_sample_indices_: return;
            for v in np.nonzero(G[u])[0]:
                if not visited[v]: dfs(v, k)

        for u in self.core_sample_indices_:
            if not visited[u]:
                dfs(u, k)
                k += 1
        #Copy of each core sample
        self.components_ = np.copy(X[self.core_sample_indices_])
        self.labels_ = labels #Noise will be labeled with -1
