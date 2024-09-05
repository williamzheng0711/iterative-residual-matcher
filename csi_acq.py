import numpy as np
from tqdm import tqdm
import cvxpy as cp
import copy

# Define your matrix dimensions
m, N = 2000, 200000
K = 100
K_tot = 300

# np.random.seed(1111)
H = np.random.normal(scale=1/np.sqrt(m), size=(m, N))
V = 10
beta_tot =  np.random.rayleigh(scale=5, size=K_tot)
active_beta_idx = np.sort(np.random.choice(K_tot, K, replace=False))
# beta = 5 * np.ones(shape=(K))
# sort the vector from largest to smallest
beta_tot = np.sort(beta_tot)[::-1]
beta = beta_tot[active_beta_idx]
print("These are the channels: ", beta, V * np.linalg.norm(beta,2)**2/m )
print(np.sort(active_beta_idx))
# generating the received signal y
# z = np.random.normal(scale=1, size=m)
z = 0
# chosenNums = np.random.choice(N, K, replace=False)
# chosenNums = np.sort(chosenNums)
chosenNums = np.arange(K)
# print("These are the index of chosen cdwds: ", chosenNums)
y = np.sqrt(V) * H[:, chosenNums] @ beta + z

s2 = np.linalg.norm(y,2)**2 / (m)
# s2 = V * np.linalg.norm(beta,2)**2/m

print("s2: ", s2)

estimated_idx = []
for j in tqdm(range(K_tot - K)): 
    u = cp.Variable((K_tot), nonneg=True)
    objective = cp.Minimize( cp.norm( ( V/m * beta_tot.reshape(1,-1)**2 @ u) -s2) )
    constraints = [cp.sum(u)== K, u[estimated_idx]==0, u<=1] if len(estimated_idx) >0 else [cp.sum(u)== K, u<=1]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    # print(u.value)
    u.value[estimated_idx] = np.Infinity
    estimated_idx.append(np.argmin(u.value))

estimated_actives = np.setdiff1d(np.arange(K_tot), estimated_idx)
indicator = np.zeros(K_tot, dtype=int)
indicator[estimated_actives] = 1
print( "mocked up: ", 1 + V/m * beta_tot.reshape(1,-1)**2 @  indicator) 

print(estimated_actives)
print( np.setdiff1d(active_beta_idx, estimated_actives) )