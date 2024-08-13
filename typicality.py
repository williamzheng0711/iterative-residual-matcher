import numpy as np
from tqdm import tqdm

# Define your matrix dimensions
m, N = 750, 100000
K = 100

H = np.random.normal(scale=1/np.sqrt(m), size=(m, N))
V = 500
beta = np.random.rayleigh(scale=5, size=K)
# beta = 5 * np.ones(shape=(K))
# sort the vector from largest to smallest
beta = np.sort(beta)[::-1]
print("These are the channels: ", beta)

# generating the received signal y
z = np.random.normal(scale=1, size=m)
# chosenNums = np.random.choice(N, K, replace=False)
# chosenNums = np.sort(chosenNums)
chosenNums = np.arange(K)
print("These are the index of chosen cdwds: ", chosenNums)
y = np.sqrt(V) * H[:, chosenNums] @ beta + z
y = np.expand_dims(y, axis=-1)

flag = True
j = 0
decodedMsgs = []
num_decoded = 0

for j in tqdm(range(K)):
    # print(y.shape)
    ress = y - beta[j]*np.sqrt(V)*H
    s2 = 1 + V/m*np.linalg.norm(beta[j:],2)**2
    temp = 1/2*np.log(2*np.pi*s2) + 1/(2*m*s2) * np.linalg.norm(ress, axis=0)**2
    entropy = 1/2 + 1/2*np.log(2*np.pi) + 1/2*np.log(s2) 
    scores = np.abs( temp - entropy )

    if len(decodedMsgs) > 0: 
        scores[decodedMsgs] = np.Infinity
    suspect = np.argsort(scores)[0]
    decodedMsgs.append(suspect)
    y = y - np.array(H[:,suspect] * beta[j] * np.sqrt(V)).reshape(m,1)
    j = j + 1


# print(P.value)
print(decodedMsgs)
losts = np.setdiff1d( decodedMsgs, chosenNums)
wrong = len(losts)
print("Accuracy: %f " %  (1-wrong/K) )