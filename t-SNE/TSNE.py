import numpy as np
from scipy.spatial import distance_matrix

class TSNE:
    def __init__(self, n_components = 2, perplexity = 30, learning_rate = 200, n_iter = 1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X):
        N = len(X)
        #Squared distance matrix
        D2 = distance_matrix(X, X) ** 2
        def Kullback_Leibler(P, Q):
            return np.sum(P[i][j] * np.log(P[i][j] / Q[i][j]) for j in range(N) for i in range(N) if i != j)
        #Calculate The perplexity of the probability distribution Pi
        def Perp(Pi):
            #Log2(0) is undefined so ignored 0
            entropy = - sum(p * np.log2(p) for p in Pi if p)
            return 2 ** entropy
        #
        def Gaussian(i, gamma):
            Pi = np.exp(- D2[i] / (2 * gamma ** 2))
            #Set P(i|i) = 0 since we're only interested in pairwise similarity
            Pi[i] = 0
            Pi /= np.sum(Pi)
            return Pi
        #Find the best gamma for x_i
        def Binary_Search(i, tol = 1e-7, n_iter = 100, lower = 1e-20, upper = 10000):
            for _ in range(n_iter):
                mid = (lower + upper) / 2
                Pi = Gaussian(i, mid)
                perplexity = Perp(Pi)
                if perplexity > self.perplexity:
                    upper = mid
                else:
                    lower = mid
                if abs(perplexity - self.perplexity) <= tol: break
            return mid

        P = np.zeros((N,N))
        for i in range(N):
            gamma = Binary_Search(i)
            P[i] = Gaussian(i, gamma)

        P = (P + P.T) / (2 * N) #ymmetrize P
        #Initial random configuration of low-dimensional space
        Y = np.random.rand(N, self.n_components)
        for _ in range(self.n_iter):
            #T-distribution Matrix :T[i,j] = (1 + ||x_i - x_j|| ** 2) ** -1
            T = 1 / (1 + distance_matrix(Y,Y) ** 2)
            #Q(i|i) = 0 since we're only interested in pairwise similarity
            np.fill_diagonal(T, 0.)
            #Normalize Q(i|j)
            Q = T / np.sum(T)
            Gradient = np.zeros(Y.shape)
            for i in range(N):
                strength = 4 * ((P[i] - Q[i]) * T[i]).reshape((-1,1))
                Gradient[i] = np.sum(strength * (Y[i] - Y), axis = 0)
            Y -= self.learning_rate * Gradient

        self.embedding_ = Y
        #Caluclate Final Kullback Leibler Divergence
        #self.Kullback_Leibler_ = Kullback_Leibler(P, Q)

        return self.embedding_
