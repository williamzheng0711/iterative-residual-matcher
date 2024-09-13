import cvxpy as cp
import numpy as np

# Define your matrix dimensions
m, N = 256, 8192
K = 100

H = np.random.rand(m, N)
beta = np.random.exponential(scale=15, size=K)
# sort the vector from largest to smallest
beta = np.sort(beta)[::-1]
print(beta)

# generating the received signal y
# z = np.random.rand(m, 1)
z = 0
y = H[:,0:K] @ beta + z

flag = True
j = 0

# Define your variable matrix P
P = cp.Variable((N, K), nonneg=True)
objective = cp.Minimize( cp.norm(y - H @ P @ beta, 2) + np.ones((1,N)) @ P @ np.ones((K,1))  )
constraints = [cp.sum(P, axis=0) >= 1, cp.sum(P, axis=1) <= 1]
problem = cp.Problem(objective, constraints)

# assert problem.is_dqcp()
# problem.solve(qcp=True)
problem.solve(verbose = True)
# problem.solve()


u = np.zeros(N)
u[ np.argsort(np.sum(P.value, axis=1))[0: N-K] ] = 1

# print(j, 1-u, problem.value)
if np.array_equal(u, u) or problem.value < 1e-7: 
    flag = False

# Print the optimal value and optimal matrix P
# print(np.setdiff1d(np.array(range(K)), ))
print(P.value)
print(np.nonzero(1-u)[0])
print(np.count_nonzero( np.nonzero(1-u)[0] - np.arange(K)))