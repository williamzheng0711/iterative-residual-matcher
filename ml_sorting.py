import numpy as np
from tqdm import tqdm
import copy

# Define your matrix dimensions
m, N = 700, 200000
K = 100

H = np.random.normal(scale=1/np.sqrt(m), size=(m, N))
V = 10
beta = np.random.rayleigh(scale=5, size=K)
# beta = 5 * np.ones(shape=(K))
# sort the vector from largest to smallest
beta = np.sort(beta)[::-1]
print("These are the channels: ", beta, beta[0]**2, np.linalg.norm(beta[1:],2)**2)

# generating the received signal y
z = np.random.normal(scale=1, size=m)
# z = 0
# chosenNums = np.random.choice(N, K, replace=False)
# chosenNums = np.sort(chosenNums)
chosenNums = np.arange(K)
print("These are the index of chosen cdwds: ", chosenNums)
y_ml_or = np.sqrt(V) * H[:, chosenNums] @ beta + z
y_ml_or = np.expand_dims(y_ml_or, axis=-1)
y_ml = copy.deepcopy(y_ml_or)


########### ML-SIC ##############
decodedMsgsML = []
num_decoded_ml = 0

for j in tqdm(range(K)):
    # print(y.shape)
    ress = y_ml - beta[j]*np.sqrt(V)*H
    temp = np.linalg.norm(ress, axis=0)**2
    scores = np.abs( temp )

    if len(decodedMsgsML) > 0: 
        scores[decodedMsgsML] = np.Infinity
    suspect = np.argsort(scores)[0]
    decodedMsgsML.append(suspect)
    y_ml = y_ml - np.array(H[:,suspect] * beta[j] * np.sqrt(V)).reshape(m,1)
    j = j + 1

# print(P.value)
print(decodedMsgsML)
lostsML = np.setdiff1d( decodedMsgsML, chosenNums)
wrongML = len(lostsML)
print("Vanilla ML-SIC Accuracy: %f " %  (1-wrongML/K) )


y_ml = y_ml_or

########### ML-SIC with sorting ##############
decodedMsgs = []
num_decoded_ml = 0

print('\x1b[7h', end='')
for j in range(K):

    ress = y_ml - beta[j]*np.sqrt(V)*H
    temp = np.linalg.norm(ress, axis=0)**2
    scores = np.abs( temp )

    if len(decodedMsgs) > 0: 
        scores[decodedMsgs] = np.Infinity

    suspect = np.argsort(scores)[0]
    decodedMsgs.append(suspect)
    y_ml = y_ml - np.array(H[:,suspect] * beta[j] * np.sqrt(V) ).reshape(m,1)

    obj_curr = np.linalg.norm(y_ml,2)
    if j > 0: # We want to sort before further decode. 
        for increment in np.sort(np.arange(start=1, step=1, stop=min(len(decodedMsgs)-1, 15),dtype=int))[::-1]:
            num_iter = 0
            flag = False
            while True: 
                num_iter += 1
                for a in range(len(decodedMsgs) - increment):
                    old_obj = obj_curr
                    b = a + increment
                    proposed_residual = y_ml + np.sqrt(V)* np.array(beta[a]*H[:,decodedMsgs[a]]+beta[b]*H[:,decodedMsgs[b]]-beta[a]*H[:,decodedMsgs[b]]-beta[b]*H[:,decodedMsgs[a]]).reshape(m,1)
                    proposed_obj = np.linalg.norm( proposed_residual, 2)
                    if proposed_obj < obj_curr: 
                        flag == True
                        temp = decodedMsgs[a]
                        decodedMsgs[a] = decodedMsgs[b]
                        decodedMsgs[b] = temp
                        obj_curr = proposed_obj
                        y_ml = proposed_residual
            
                if flag == False or num_iter == 200:
                    # print("Reordering phase %d finished, used %d many iterations" % (increment ,num_iter))
                    break
    print(decodedMsgs, end="\r", )
    j = j + 1
# print('\x1b[7h', end='')


print(decodedMsgs)
lostsML = np.setdiff1d(decodedMsgs, chosenNums)
wrongML = len(lostsML)
print("ML-SIC with Sorting Accuracy: %f " %  (1-wrongML/K) )