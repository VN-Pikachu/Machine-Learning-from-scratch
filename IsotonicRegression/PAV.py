from collections import deque
import numpy as np
class PVA:
    def fit(self, A, W):
        A, W = np.array(A), np.array(W)
        N = len(A)
        self.solution_ = np.zeros(N)
        #The i-th SolutionBlock is an interval [l(i), r(i)] (l(i) <= r(i))
        #For each SolutionBlock, the solution y for that interval is the same
        #    e.g: SolutionBlock : interval([1,3]) => solution: y[1] = y[2] = y[3] = constant
        #The i-th SolutionBlock is represented by l(i)
        #    e.g: SolutionBlock: interval([2,5]) => index 2 represents this SolutionBlock

        Q1 = deque(range(N))
        #Foward and Backward array function like double linklist
        #Foward array:F[i]:The index of the next block of the i-th block
        F = list(range(1, N + 1))
        #Backward array:B[i]:The index of the previous block of the i-th block
        B = list(range(-1, N))
        #the solution y for the i-th block: y = weighted average of the interval [i, F[i]]
        def solution(i):
            a, w = A[i:F[i]], W[i:F[i]]
            return a @ w / sum(w)


        while Q1:
            Q2 = deque()
            #Marker: Mark the furthest SolutionBlock we have considered so far
            marker = 0
            while Q1:
                #Pop a SolutionBlock from the queue
                cur = Q1.popleft()
                #Pooled = True if adjacent violators occurs
                pooled = False
                if cur >= marker:
                    #As long as 2 consecutive SolutionBlocks are Adjacent-violators:
                    while F[cur] != N and solution(cur) >= solution(F[cur]):
                        pooled = True
                        #Merge 2 adjacent SolutionBlocks
                        #Just like merging 2 consective nodes of a DoubleLinklist
                        next = F[F[cur]]
                        F[cur] = next
                        if next != N:B[next] = cur
                        #Move Marker to the next SoluionBlock of the current SolutionBlock (after merged)
                        marker = next #This is equivalent to: marker = F[cur]
                #If Adjacent-violators occurs for the current SolutionBlock (the one popped from the Queue)
                #It means that we have merged the current SolutionBlock
                #With at least 1 SolutionBlock after the current SolutionBlock
                #The current SolutionBlock after being merged becomes a Larger SolutionBlock
                #The solution y over the new merged interval of the current SolutionBlock changes
                #So it's solution y might be bigger the the one of it's previous SolutionBlock
                #So we need to add the previous SolutionBlock of the current SolutionBlock
                #into Q2 for latter consideration
                if pooled and B[cur] != -1: Q2.append(B[cur])
            #Interchange Q1 and Q2 to consider Candidate SolutionBlocks that might need to change
            Q1 = Q2

        tmp = 0
        while tmp != N:
            self.solution_[tmp:F[tmp]] = solution(tmp)
            tmp = F[tmp]

        self.error_ = W @ ((self.solution_ - A) ** 2)
