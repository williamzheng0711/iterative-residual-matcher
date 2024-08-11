import cvxpy as cp
import numpy as np
from copy import deepcopy

# Define your matrix dimensions
m, N = 256, 15000
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
decodedMsgs = []
num_decoded = 0
beta_or = deepcopy(beta)

while True:
    # Define your variable matrix P
    P = cp.Variable((N, K-num_decoded), nonneg=True)
    objective = cp.Minimize( cp.norm(y - H @ P @ beta, 2) + np.ones((1,N)) @ P @ np.ones((K-num_decoded,1))  )
    constraints = [cp.sum(P, axis=0) >= 1, 
                   cp.sum(P, axis=1) <= 1, 
                   P[decodedMsgs,:] == 0, ] if len(decodedMsgs)>0  else [cp.sum(P, axis=0) >= 1, 
                                                                         cp.sum(P, axis=1) <= 1,]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose = True)

    u = np.zeros(N)
    u[ np.argsort(np.sum(P.value, axis=1))[0: N-(K -num_decoded)] ] = 1
    u = 1 - u # suspects are indicated by 1, others are indicated by 0
    u = np.array(u, dtype=int)
    suspects = np.nonzero(u)[0]
    print("These are the most suspects: ", suspects)
    rnd_correct = K - num_decoded - np.count_nonzero(suspects - np.arange(num_decoded, K))
    if rnd_correct == 0:
        break
    y = y - H[:,suspects[0:rnd_correct]] @ [ beta_or[j] for j in range(num_decoded, num_decoded + rnd_correct) ]
    # y_remain_true = H[:, range(num_decoded + rnd_correct,K)] @ beta[-(K-(num_decoded+rnd_correct)):]
    # print(np.linalg.norm(y - y_remain_true,2), "hahahahaha")

    num_decoded += rnd_correct
    beta = beta[-(K-num_decoded):]
    decodedMsgs = decodedMsgs + np.nonzero(u)[0][0:rnd_correct].tolist()
    print(j, u, rnd_correct, len(decodedMsgs))
    if len(decodedMsgs) == K:
        break

    j = j + 1

# print(j, 1-u, problem.value)
if np.array_equal(u, u) or problem.value < 1e-7: 
    flag = False

print(P.value)
print(decodedMsgs)
# print(np.count_nonzero( np.nonzero(u)[0] - np.arange(K)))