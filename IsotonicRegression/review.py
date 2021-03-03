from collections import deque
import numpy as np
class PVA:
    def fit(self, A, W):
        A, W  = np.array(A), np.array(W)
        N = len(A)
        F, B = list(range(1,N+1)), list(range(-1,N))
        def solution(i):
            a, w = A[i:F[i]], W[i:F[i]]
            return a @ w / np.sum(w)
        Q1 = deque(np.arange(N))
        while Q1:
            Q2 = deque()
            marker = 0
            while Q1:
                k = Q1.popleft()
                pooled = 0
                if k < marker:continue
                while F[k] != N and solution(k) >= solution(F[k]):
                    u = F[F[k]]
                    F[k] = u
                    marker = u
                    if u != N:B[u] = k
                    pooled = 1
                if pooled and B[k] != -1: Q2.append(B[k])
            Q1 = Q2
        self.solution_ = np.zeros(N)
        i = 0
        while i != N:
            j = F[i]
            self.solution_[i:j] = solution(i)
            i = j
        self.error_ = (self.solution_ - A) ** 2 @ W
