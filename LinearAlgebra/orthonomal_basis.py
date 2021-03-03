 def orth(M):
        #M is a matrix with vectors as rows
        def vector(v):return np.array(v).reshape((-1,1))
        #project vector v onto vector a
        def p(v, a):
            v, a = vector(v), vector(a)
            return (a @ np.linalg.inv(a.T @ a) @ a.T @ v).flatten()
        m, n = M.shape
        #basis is a matrix with rows are orthonomal basis vectors
        basis = np.copy(M)
        for i in range(m):
            for j in range(i):
                basis[i] -= p(M[i], basis[j])
        #normalize each vector to unit length
        length = np.linalg.norm(basis, axis = 1)
        basis /= vector(length)
        return basis