import numpy as np
from tqdm import tqdm
import copy

# Define your matrix dimensions
m, N = 800, 100000
K = 100

H = np.random.normal(scale=1/np.sqrt(m), size=(m, N))
V = 1
beta = np.random.rayleigh(scale=10, size=K)
# beta = 5 * np.ones(shape=(K))
# sort the vector from largest to smallest
beta = np.sort(beta)[::-1]
print("These are the channels: ", beta, beta[0]**2, np.linalg.norm(beta[1:],2)**2 )

# generating the received signal y
# z = np.random.normal(scale=1, size=m)
z = 0
# chosenNums = np.random.choice(N, K, replace=False)
# chosenNums = np.sort(chosenNums)
chosenNums = np.arange(K)
print("These are the index of chosen cdwds: ", chosenNums)
y_tpk = np.sqrt(V) * H[:, chosenNums] @ beta + z
y_tpk = np.expand_dims(y_tpk, axis=-1)
y_ml = copy.deepcopy(y_tpk)
y_tm = copy.deepcopy(y_tpk)
y_omp = copy.deepcopy(y_tpk)

tpk_dec = False
ml_dec = True
tpk_ml_dec = False
omp_dec = True


# if tpk_dec == True:
# ########### Typicality Decoding ##############
#     decodedMsgs = []
#     num_decoded = 0

#     for j in tqdm(range(K)):
#         # print(y.shape)
#         ress = y_tpk - beta[j]*np.sqrt(V)*H
#         s2 = 1 + V/m*np.linalg.norm(beta[j:],2)**2
#         temp = 1/2*np.log(2*np.pi*s2) + 1/(2*m*s2) * np.linalg.norm(ress, axis=0)**2
#         entropy = 1/2 + 1/2*np.log(2*np.pi) + 1/2*np.log(s2) 
#         scores = np.abs( temp - entropy )

#         if len(decodedMsgs) > 0: 
#             scores[decodedMsgs] = np.Infinity
#         suspect = np.argsort(scores)[0]
#         decodedMsgs.append(suspect)
#         y_tpk = y_tpk - np.array(H[:,suspect] * beta[j] * np.sqrt(V)).reshape(m,1)
#         j = j + 1

#     # print(P.value)
#     print(decodedMsgs)
#     losts = np.setdiff1d( decodedMsgs, chosenNums)
#     wrong = len(losts)
#     print("Accuracy: %f " %  (1-wrong/K) )


if ml_dec == True: 
    ########### ML Decoding ##############
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
    print("ML Accuracy: %f " %  (1-wrongML/K) )


# if tpk_ml_dec == True: 
# ########### Typicality & ML Decoding ##############
#     decodedMsgsTM = []
#     num_decodedTM = 0

#     for j in tqdm(range(K)):
#         ## TPK part
#         ressTM = y_tm - beta[j]*np.sqrt(V)*H
#         s2 = 1 + V/m*np.linalg.norm(beta[j:],2)**2
#         tempTPK = 1/2*np.log(2*np.pi*s2) + 1/(2*m*s2) * np.linalg.norm(ressTM, axis=0)**2
#         entropy = 1/2 + 1/2*np.log(2*np.pi) + 1/2*np.log(s2) 
#         scoresTPK = np.abs( tempTPK - entropy )

#         if len(decodedMsgsTM) > 0: 
#             scoresTPK[decodedMsgsTM] = np.Infinity

#         suspectsTPK = np.argsort(scoresTPK)[ 0: int(np.ceil(len(scoresTPK)*0.99)) ]

#         ## ML part
#         scoresML = np.linalg.norm(ressTM, axis=0)**2
#         if len(decodedMsgsTM) > 0: 
#             scoresML[decodedMsgsTM] = np.Infinity
        
#         suspectsML = np.argsort(scoresML)
#         for suspect in suspectsML:
#             if suspect in suspectsTPK:
#                 decodedMsgsTM.append(suspect)
#                 break

#         y_tm = y_tm - np.array(H[:,suspect] * beta[j] * np.sqrt(V)).reshape(m,1)
#         j = j + 1

#     # print(P.value)
#     print(decodedMsgsTM)
#     losts = np.setdiff1d( decodedMsgsTM, chosenNums)
#     wrong = len(losts)
#     print("Accuracy: %f " %  (1-wrong/K) )


if omp_dec == True: 
########### Typicality & ML Decoding ##############
    decodedMsgsOMP = []
    num_decodedOMP = 0

    for j in tqdm(range(K)):
        ## TPK part
        scoresOMP = np.abs( y_omp.reshape(1,m) @ H ).reshape(-1)
        # print(scoresOMP)
        if len(decodedMsgsOMP) > 0: 
            scoresOMP[decodedMsgsOMP] = - np.Infinity
        
        suspectOMP = np.argmax(scoresOMP)
        decodedMsgsOMP.append(suspectOMP)

        coeff =  y_omp.reshape(1,m) @ H[:,suspectOMP]
        y_omp = y_omp - np.sqrt(V) * np.array( coeff * H[:,suspectOMP]).reshape(m,1)
        j = j + 1

    # print(P.value)
    print(decodedMsgsOMP)
    losts = np.setdiff1d( decodedMsgsOMP, chosenNums)
    wrong = len(losts)
    print("Accuracy OMP: %f " %  (1-wrong/K) )