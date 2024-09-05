import numpy as np
from tqdm import tqdm
import copy
import cvxpy as cp

K= 100
m = 700
N= 200000
V = 30
# V1 = 0

beta = np.random.rayleigh(scale=5, size=K)
beta = np.sort(beta)[::-1]

c = np.random.binomial(n=30000, p = 1/30000, size=K)
print(c)

H = np.random.normal(scale=1/np.sqrt(m), size=(m, N))

chosenNums = np.arange(K)
# print("These are the index of chosen cdwds: ", chosenNums)
z = np.random.normal(scale=1, size=m)
y_ml_or = np.sqrt(V) * H[:, chosenNums] @ beta + z

# for k, c_k in enumerate(c):
#     y_ml_or[k*int(m/K) : (k+1)*int(m/K)] += np.sqrt(V1) * c_k * beta[k] * np.sqrt(K/m)*np.ones((int(m/K)), dtype=int)

# c_hat = [max(np.sum(y_ml_or[k*int(m/K) : (k+1)*int(m/K)]) / (np.sqrt(V1)*np.sqrt(m/K)*beta[k]), 0) for k in range(K)]

# print(np.array(c_hat, dtype=int))

s2 = np.var(y_ml_or)
print( s2, 1+ V/m* np.sum(beta**2 * c))

ps = [np.sum( y_ml_or[k*int(m/K) : (k+1)*int(m/K)] ) for k in range(K)]


# c_var = cp.Variable((K), nonneg=True)

# # Define your objective function
# # objective = cp.Minimize(cp.norm(y - H*P*beta, 2) + cp.sum(cp.power(P, 0.1)) )
# objective = cp.Minimize(  (cp.norm(y_ml_or,2)**2 + V1*cp.sum(beta*c_var**2) - 2*np.sqrt(V1*m/K)* cp.sum(ps*c_var*beta)) / (m+ V1* cp.sum(beta**2*c_var)) )

# # Define your constraints
# constraints = [c_var >=0.01, c_var<=4]

# # Define your problem
# problem = cp.Problem(objective, constraints)

# # Solve your problem
# assert problem.is_dqcp()
# problem.solve(qcp=True, solver = "MOSEK", verbose=True)

print(c_var)