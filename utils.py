import numpy as np
import utils
from tqdm import tqdm


def pdf_Rayleigh(scale, x):
    return x*np.exp(-x**2 /(2*scale**2) ) / scale**2


def vSIC(y_vsic, K, beta, H, N, V, prtDetail=True): 
    ########### vanilla SIC ##############
    decodedMsgs_vsic = []

    for j in (tqdm(range(K)) if prtDetail else range(K)):
        ress = y_vsic - beta[j]*np.sqrt(V)*H
        temp = np.linalg.norm(ress, axis=0)**2
        scores = np.abs( temp )

        if len(decodedMsgs_vsic) > 0: 
            scores[decodedMsgs_vsic] = np.Infinity
        suspect = np.argsort(scores)[0]
        decodedMsgs_vsic.append(suspect)
        y_vsic = y_vsic - np.array(H[:,suspect] * beta[j] * np.sqrt(V)).reshape(N,1)
        j = j + 1
    return decodedMsgs_vsic



def IRM(y_irm, K, beta, H, N, V, prtDetail= False, midWayOut = False):
    ########### Iterative Residual Matcher ##############
    decodedMsgs_irm = []


    ## This part is to obtain an initial guess (which is itself good) to the problem
    for j in range(K):
        ress = y_irm - beta[j]*np.sqrt(V)*H
        temp = np.linalg.norm(ress, axis=0)**2
        scores = np.abs( temp )

        if len(decodedMsgs_irm) > 0: 
            scores[decodedMsgs_irm] = np.Infinity

        suspect = np.argsort(scores)[0]
        decodedMsgs_irm.append(suspect)
        y_irm = y_irm - np.array(H[:,suspect] * beta[j] * np.sqrt(V) ).reshape(N,1)

        # This part is the "pairwise switch before decoding" part
        y_irm, decodedMsgs_irm = pairwise_switch(decodedMsgs_irm, y_irm, beta, H, N, V)
        if prtDetail == True:
            print(decodedMsgs_irm, end="\r", )
        j = j + 1

    if midWayOut:
        return decodedMsgs_irm

    ## This part is to refine the previously obtained initial guess. 
    ## We try to move to a local minimum near it. 
    invariant_slot = 1
    nRounds = 0
    while invariant_slot or nRounds < 3:
        invariant_slot = 0

        # This part (roughly) ensures the invariance wrt each slot adjustment
        for k in (tqdm(np.arange(start=K-1, step=-1, stop=-1)) if prtDetail == True else np.arange(start=K-1, step=-1, stop=-1)):
            msg_k = decodedMsgs_irm[k]
            y_iter = y_irm + beta[k]*np.sqrt(V)*np.array(H[:,msg_k]).reshape(-1,1)
            ress = y_iter - beta[k]*np.sqrt(V)*H
            temp = np.linalg.norm(ress, axis=0)**2
            scores = np.abs( temp )
            excludeIndex = [a for a in decodedMsgs_irm if a != msg_k]
            scores[excludeIndex] = np.Infinity
            suspect = np.argsort(scores)[0]
            if suspect != msg_k: 
                invariant_slot = 1
                if prtDetail:
                    print(k, msg_k, suspect)
            decodedMsgs_irm[k] = suspect
            y_irm = y_irm + beta[k]*np.sqrt(V)*np.array(H[:, msg_k]).reshape(-1,1) - beta[k]*np.sqrt(V)*np.array(H[:, suspect]).reshape(-1,1)
        if prtDetail:
            print("Done with CD")
        
        # This part (roughly) ensures the invariance wrt pairwisely permutation
        y_irm, decodedMsgs_irm = pairwise_switch(decodedMsgs_irm, y_irm, beta, H, N, V)
        if prtDetail:
            print("Done with 2Perm")

        nRounds += 1
    return decodedMsgs_irm


def pairwise_switch(decodedMsgs_irm, y_irm, beta, H, N, V, prtDetail=False):
    obj_curr = np.linalg.norm(y_irm,2)
    arr = np.sort(np.arange(start=1, step=1, stop=min(len(decodedMsgs_irm)-1, 30),dtype=int))[::-1]
    for increment in (tqdm(arr) if prtDetail==True else arr):
        num_iter = 0
        flag = False
        while True: 
            num_iter += 1
            for a in range(len(decodedMsgs_irm) - increment):
                b = a + increment
                proposed_residual = y_irm + np.sqrt(V)* np.array(beta[a]*H[:,decodedMsgs_irm[a]]+beta[b]*H[:,decodedMsgs_irm[b]]-beta[a]*H[:,decodedMsgs_irm[b]]-beta[b]*H[:,decodedMsgs_irm[a]]).reshape(N,1)
                proposed_obj = np.linalg.norm( proposed_residual, 2)
                if proposed_obj < obj_curr: 
                    flag == True
                    temp = decodedMsgs_irm[a]
                    decodedMsgs_irm[a] = decodedMsgs_irm[b]
                    decodedMsgs_irm[b] = temp
                    obj_curr = proposed_obj
                    y_irm = proposed_residual
        
            if flag == False :
                break
    return y_irm, decodedMsgs_irm

def get_distanceInfo(K, chosenNums, decodedMsgs):
    distances = np.array([ np.abs(decodedMsg - chosenNums[idx]) if decodedMsg in chosenNums else 10*K for idx, decodedMsg in enumerate(decodedMsgs)], dtype=int)
    distances_toAdd = [ np.count_nonzero(distances == n) for n in range(K) ]
    # print(distances, distances_toAdd)
    return distances, distances_toAdd


def evaluate_result(K, chosenNums, decodedMsgs, rayleigh_scale, beta):
    """
    Returns accuracy and EUIM
    """
    losts = np.setdiff1d( decodedMsgs, chosenNums)
    num_wrong = len(losts)
    accuracy = 1-num_wrong/K
    EUIM     = 1/K*sum([ utils.pdf_Rayleigh(scale=rayleigh_scale, x=beta[cn]) * 
                                    (np.abs(beta[cn] - beta[decodedMsgs.index(cn)]) if cn in decodedMsgs else beta[cn])  for cn in chosenNums])
    return accuracy, EUIM
    