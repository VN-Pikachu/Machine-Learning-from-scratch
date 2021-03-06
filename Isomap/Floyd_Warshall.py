import numpy as np
def matrix(G):
    G = np.copy(G).astype('float')
    G[G == 0] = np.inf
    n = len(G)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                G[i,j] = min(G[i,j], G[i,k] + G[k,j])
    return G
