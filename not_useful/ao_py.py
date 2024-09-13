import cvxpy as cp
import numpy as np

# Define your matrix dimensions
m, N = 256, 8192
K = 100

H = np.random.rand(m, N)
beta = np.random.exponential(scale=15, size=K)
# sort the vector from largest to smallest
beta = np.sort(beta)[::-1]

# generating the received signal y
# z = np.random.rand(m, 1)
z = 0
y = H[:,0:K] @ beta + z

u = np.ones(N) 
u[np.random.choice(N, K, replace=False)] = 0

flag = True
j = 0

while flag == True: 
    # Define your variable matrix P
    P = cp.Variable((N, K), nonneg=True)
    objective = cp.Minimize(cp.norm(y - H @ P @ beta, 2) + u @ P @ np.ones((K,1) ))
    constraints = [cp.sum(P, axis=0) == 1, cp.sum(P, axis=1) <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    u_new = np.zeros(N)
    u_new[ np.argsort(np.sum(P.value, axis=1))[0: N-K] ] = 1

    print(j, 1-u_new, problem.value)
    if np.array_equal(u_new, u) or problem.value < 1e-7: 
        flag = False
    u = u_new
    j = j + 1

# Print the optimal value and optimal matrix P
# print(np.setdiff1d(np.array(range(K)), ))
print(P.value)
print(np.nonzero(1-u)[0])
print(np.count_nonzero( np.nonzero(1-u)[0] - np.arange(K)))