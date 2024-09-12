import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import copy
import matplotlib.pyplot as plt
from optparse import OptionParser
# from joblib import Parallel, delayed
# from parallel import ParallelTqdm


def pdf_Rayleigh(scale, x):
    return x*np.exp(-x**2 /(2*scale**2) ) / scale**2



def lauch_once_greedySIC(y_ml, m, K,V, rayleigh_scale):
    ########### ML Decoding ##############
    decodedMsgsML = []

    for j in range(K):
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

    toAdd = [decodedMsg in chosenNums for decodedMsg in decodedMsgsML]


    toAdd2 = 1/K*sum([ pdf_Rayleigh(scale=rayleigh_scale, x=beta[cn]) * (np.abs(beta[cn] - beta[decodedMsgsML.index(cn)]) if cn in decodedMsgsML else beta[cn])  for cn in chosenNums])

    distances = np.array([ np.abs(decodedMsg - chosenNums[idx]) if decodedMsg in chosenNums else 10*K for idx, decodedMsg in enumerate(decodedMsgsML)], dtype=int)
    distances_toAdd = [ np.count_nonzero(distances == n) for n in range(K) ]
    # return [ np.count_nonzero(distances == n) for n in range(K) ]

    return toAdd, toAdd2, distances_toAdd


def lauch_once_greedySICsort(y_ml, m, K,V, rayleigh_scale):
    ########### ML Decoding ##############
    decodedMsgs = []

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
            arr = np.sort(np.arange(start=1, step=1, stop=min(len(decodedMsgs)-1, 20),dtype=int))[::-1]
            # print(arr)
            for increment in arr:
                num_iter = 0
                flag = False
                while True: 
                    num_iter += 1
                    for a in range(len(decodedMsgs) - increment):
                        # old_obj = obj_curr
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
                
                    if flag == False :
                        # print("Reordering phase %d finished, used %d many iterations" % (increment ,num_iter))
                        break
        j = j + 1

    toAdd = [decodedMsg in chosenNums for decodedMsg in decodedMsgs]

    toAdd2 = 1/K*sum([ pdf_Rayleigh(scale=rayleigh_scale, x=beta[cn]) * (np.abs(beta[cn] - beta[decodedMsgs.index(cn)]) if cn in decodedMsgs else beta[cn])  for cn in chosenNums])

    distances = np.array([ np.abs(decodedMsg - chosenNums[idx]) if decodedMsg in chosenNums else 10*K for idx, decodedMsg in enumerate(decodedMsgs)], dtype=int)
    distances_toAdd = [ np.count_nonzero(distances == n) for n in range(K) ]

    return toAdd, toAdd2, distances_toAdd



## Accept user inputs, specifying simulation arguments
parser = OptionParser()
parser.add_option("--args", type="string", dest="args", help="Arguments", default="")
parser.add_option("--K", type="int", dest="K", help="Number of users", default=-1)
parser.add_option("--m", type="int", dest="m", help="Number of channel use", default=-1)
parser.add_option("--N", type="int", dest="N", help="width of sensing matrix", default=-1)
parser.add_option("--RS", type="int", dest="RayleighScale", help="RayleighScale", default=-1)
parser.add_option("--noisePower", type="float", dest="noisePower", help="noisePower", default=-1)
parser.add_option("--num_trials", type="int", dest="num_trials", help="num of trials", default=-1)

(options, args) = parser.parse_args()

print( " --- --- --- --- --- ")

### Examine whether the user inputs are valid
K = options.K;                                  assert K > 0 
m = options.m;                                  assert m > 100
N = options.N;                                  assert N > m 
RayleighScale = options.RayleighScale;          assert RayleighScale > 0
noisePower = options.noisePower;                assert noisePower >= 0
num_trials = options.num_trials;                assert num_trials > 0

print("K, RayleighScale, noisePower, num_trials: ", K, RayleighScale, noisePower, num_trials)

V = 1

np.random.seed(8)

active_freq_total1 = np.zeros(K, dtype=int)
active_freq_total2 = np.zeros(K, dtype=int)
match_loss_total1 = 0
match_loss_total2 = 0
distance_total1 = np.zeros(K, dtype=int)
distance_total2 = np.zeros(K, dtype=int)

for j in tqdm(range(num_trials)):
    H = np.random.normal(scale=1/np.sqrt(m), size=(m, N))
    beta = np.random.rayleigh(scale=RayleighScale, size=K)
    # beta = 5 * np.ones(shape=(K))
    # sort the vector from largest to smallest
    beta = np.sort(beta)[::-1]
    # print("These are the channels: ", beta, beta[0]**2, np.linalg.norm(beta[1:],2)**2 )

    # generating the received signal y
    # z = np.random.normal(scale=1, size=m)
    z = 0 if noisePower == 0 else np.random.normal(scale=noisePower, size=m)
    # chosenNums = np.random.choice(N, K, replace=False)
    # chosenNums = np.sort(chosenNums)
    chosenNums = np.arange(K)
    # print("These are the index of chosen cdwds: ", chosenNums)
    y_ml = np.sqrt(V) * H[:, chosenNums] @ beta + z
    y_ml = np.expand_dims(y_ml, axis=-1)

    y_ml_1 = copy.deepcopy(y_ml)
    active_freq_1_toadd, match_freq_toadd1, distance_toadd1 = lauch_once_greedySIC(y_ml_1, m, K, V, RayleighScale)
    
    active_freq_total1 += np.array(active_freq_1_toadd)
    match_loss_total1  += match_freq_toadd1
    distance_total1 += distance_toadd1

    y_ml_2 = copy.deepcopy(y_ml)
    freq_2_toadd, match_freq_toadd2, distance_toadd2 = lauch_once_greedySICsort(y_ml_2, m, K, V, RayleighScale)
    active_freq_total2 += np.array(freq_2_toadd)
    match_loss_total2  += match_freq_toadd2
    distance_total2 += distance_toadd2

print(" --- Vanilla ---")
print("Accuracy: ", (np.sum(active_freq_total1)/ num_trials) / K)
print("Num trials, match loss: ", num_trials, match_loss_total1)
print("Active shift distance: ", distance_total1)
print(" --- ")

print(" --- IRM ---")
print("Accuracy: ", (np.sum(active_freq_total2)/ num_trials) / K)
print("Num trials, match loss: ", num_trials, match_loss_total2)
print("Active shift distance: ", distance_total2)
print(" --- ")

# plt.plot(range(K), 1/num_trial * match_freq_total1)
# plt.plot(range(K), 1/num_trial * match_freq_total2)
# plt.plot(range(K), 1/num_trial * match_freq_total3)
# plt.plot(range(K), 1/num_trial * match_freq_total4)

# plt.legend(["vanilla noise0.1", "IRM noise.1", "vanilla noise.3", "IRM noise.3"])
# plt.show()