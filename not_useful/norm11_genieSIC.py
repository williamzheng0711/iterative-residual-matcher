import cvxpy as cp
import numpy as np
from copy import deepcopy

# Define your matrix dimensions
m, N = 40, 512
K = 40

H = np.random.normal(scale=1/np.sqrt(m), size=(m, N))
beta = np.random.rayleigh(scale=5, size=K)
# sort the vector from largest to smallest
beta = np.sort(beta)[::-1]
V = 1
print("These are the channels: ", beta)

# generating the received signal y
z = 0
# z = 0.1 * np.random.rand(m)
# chosenNums = np.random.choice(N, K, replace=False)
# chosenNums = np.sort(chosenNums)
chosenNums = range(0,K)
y = np.sqrt(V) * H[:, chosenNums] @ beta + z
print("These are the index of chosen cdwds: ", chosenNums)

flag = True
j = 1
decodedMsgs = []
num_decoded = 0
beta_or = deepcopy(beta)

while True:
    # Define your variable matrix P
    P = cp.Variable((N, K-num_decoded), nonneg= True)
    objective = cp.Minimize( cp.norm(y - np.sqrt(V) * H @ P @ beta, 2) + np.ones((1,N)) @ P @ np.ones((K-num_decoded,1))  )
    constraints = [ P >=0, 
                   cp.sum(P, axis=0) >= 1, 
                   cp.sum(P, axis=1) <= 1, 
                   P[decodedMsgs,:] == 0, ] if len(decodedMsgs)>0  else [cp.sum(P, axis=0) >= 1, 
                                                                         cp.sum(P, axis=1) <= 1,]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()

    u = np.ones(N)
    likelihoods_ascend = np.argsort(np.sum(P.value, axis=1))
    unlikely = likelihoods_ascend[0: N-(K -num_decoded)]
    mostLikely = likelihoods_ascend[N-(K -num_decoded) : ]
    u[ unlikely ] = 0  # suspects are indicated by 1, others are indicated by 0
    u = np.array(u, dtype=int)
    suspects = mostLikely[::-1]
    print("- - - Decoding round %d starts - - - " % j)
    print("- - - - - - These are the most likely suspects: ", suspects)
    rnd_correct = K - num_decoded - np.count_nonzero( np.cumsum(np.abs(suspects - chosenNums[num_decoded: K])) )
    
    if rnd_correct == 0:
        y = y - H[:,suspects[0:1]] @ [ beta_or[j] for j in range(num_decoded, num_decoded + 1) ]
        num_decoded += 1
        beta = beta[-(K-num_decoded):]
        decodedMsgs = decodedMsgs + np.nonzero(u)[0][0:1].tolist()

    elif rnd_correct > 0:
        y = y - H[:,suspects[0:rnd_correct]] @ [ beta_or[j] for j in range(num_decoded, num_decoded + rnd_correct) ]
        # y_remain_true = H[:, range(num_decoded + rnd_correct,K)] @ beta[-(K-(num_decoded+rnd_correct)):]
        # print(np.linalg.norm(y - y_remain_true,2), "hahahahaha")

        num_decoded += rnd_correct
        beta = beta[-(K-num_decoded):]
        decodedMsgs = decodedMsgs + np.nonzero(u)[0][0:rnd_correct].tolist()
        # print(j, u, rnd_correct, len(decodedMsgs))
    
    print("- - - - - - In round %d, we corrected: %d cdwds" % (j, rnd_correct if rnd_correct>0 else 1) )

    if len(decodedMsgs) == K:
        break
    
    j = j + 1

# print(j, 1-u, problem.value)
if np.array_equal(u, u) or problem.value < 1e-7: 
    flag = False

# print(P.value)
print("Finally, we decoded: ", decodedMsgs)
print(len(np.setdiff1d( decodedMsgs, np.arange(K))))