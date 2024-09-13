import cvxpy as cp
import numpy as np

# Define your matrix dimensions
m, N = 256, 2048
K = 5

H = np.random.rand(m, N)
beta = np.random.exponential(scale=15, size=K)
# sort the vector from largest to smallest
beta = np.sort(beta)[::-1]

# generating the received signal y
# z = np.random.rand(m, 1)
z = 0
y = np.matmul(H[:,0:K], beta) + z

# Define your variable matrix P
P = cp.Variable((N, K), nonneg=True)

# Define your objective function
# objective = cp.Minimize(cp.norm(y - H*P*beta, 2) + cp.sum(cp.power(P, 0.1)) )
objective = cp.Minimize(cp.norm(y - H*P*beta, 2)**2 )

# Define your constraints
constraints = [cp.sum(P, axis=0) == 1]

# Define your problem
problem = cp.Problem(objective, constraints)

# Solve your problem
assert problem.is_dqcp()
problem.solve(qcp=True)

# Print the optimal value and optimal matrix P
print("Optimal value: ", problem.value)
print("Optimal matrix P: ")
print(P.value)

# print(cp.__version__)
# print(cp.installed_solvers())