import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import floyd_warshall
from Floyd_Warshall import matrix
class Isomap:
    def __init__(self, n_neighbors = 5, n_components = 2):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
    def fit(self, X):
        n = len(X)
        #Build k-neighbors Graph
        Graph = kneighbors_graph(X, n_neighbors = self.n_neighbors, mode = 'distance').toarray()
        #APSP Graph
        D = matrix(Graph)
        #D = floyd_warshall(Graph, directed = False)
        #Apply MDS on APSP Graph
        H = np.eye(n) - np.ones((n,n)) / n
        G = -.5 * H @ (D ** 2) @ H
        U, S, V = np.linalg.svd(G)
        X = U @ np.diag(S ** .5)
        self.embedding_ = X[:, :self.n_components]
        return self.embedding_
