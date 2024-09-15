import numpy as np
from tqdm import tqdm
import copy
from utils import * 

### Some reminders 
# We use "M" denotes the "2^B" in the paper, since we can choose M as any integer. 

np.random.seed(14)

N, M = 800, 200000
K = 125
rayleigh_scale = 8
noise_power = 0.1
H = np.random.normal(scale=1/np.sqrt(N), size=(N, M))  

V = 1 

# Generating the channel coefficients, and order them from the largest to the smallest. 
beta = np.random.rayleigh(scale=rayleigh_scale, size=K)
beta = np.sort(beta)[::-1]
print("The are the channel coefficients: ", beta)

# Generating the received signal y
z = np.random.normal(scale=noise_power, size=N)
chosenNums = np.arange(K)   ## For simplicity and WLOG, one can always TX the first K messages. With message k pairing with h_k
print("These are the index of chosen cdwds: ", chosenNums)
y_or = np.sqrt(V) * H[:, chosenNums] @ beta + z
y_or = np.expand_dims(y_or, axis=-1)



########### vanilla SIC ##############
y_vsic = copy.deepcopy(y_or)
decodedMsgs_vsic = vSIC(y_vsic=y_vsic, K=K, beta=beta, H=H, N=N, V=V, prtDetail=True)
print(decodedMsgs_vsic)
accuracy_vsic, EUIM_vsic = evaluate_result(K=K, chosenNums=chosenNums, decodedMsgs=decodedMsgs_vsic, rayleigh_scale=rayleigh_scale, beta=beta)
print(" vanilla SIC Accuracy: %f " %  accuracy_vsic)
print(" vanilla SIC EUIM: " + str(EUIM_vsic) )



########### Iterative Residual Matcher ##############
y_irm = copy.deepcopy(y_or)
decodedMsgs_irm = IRM(y_irm=y_irm, K=K, beta=beta, H=H, N=N, V=V, prtDetail=True)
accuracy_irm, EUIM_irm = evaluate_result(K=K, chosenNums=chosenNums, decodedMsgs=decodedMsgs_irm, rayleigh_scale=rayleigh_scale, beta=beta)
print(" IRM Accuracy: %f " %  accuracy_irm)
print(" IRM EUIM: " + str(EUIM_irm))

distances, distances_toAdd= get_distanceInfo(K, chosenNums, decodedMsgs_irm)
print(" Mismatch frequency: ", distances_toAdd)