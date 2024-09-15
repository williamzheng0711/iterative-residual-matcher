import numpy as np
import contextlib
import copy
import joblib
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm import tqdm
from optparse import OptionParser
from tqdm.contrib.concurrent import process_map
from utils import *


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def single_vsic_and_irm(N, M, K, V, rayleigh_scale, noisePower):
    H = np.random.normal(scale=1/np.sqrt(N), size=(N, M))
    beta = np.random.rayleigh(scale=rayleigh_scale, size=K)
    beta = np.sort(beta)[::-1]

    # generating the received signal y
    z = 0 if noisePower == 0 else np.random.normal(scale=noisePower, size=N)
    chosenNums = np.arange(K)
    y_or = np.sqrt(V) * H[:, chosenNums] @ beta + z
    y_or = np.expand_dims(y_or, axis=-1)

    y_1 = copy.deepcopy(y_or)
    decodedMsgs_vsic = vSIC(y_vsic=y_1, K=K, beta=beta, H=H, N=N, V=V, prtDetail=False)
    accuracy_vsic, EUIM_vsic = evaluate_result(K=K, chosenNums=chosenNums, decodedMsgs=decodedMsgs_vsic, rayleigh_scale=rayleigh_scale, beta=beta)
    _, distances_vsic_toAdd= get_distanceInfo(K, chosenNums, decodedMsgs_vsic)

    y_2 = copy.deepcopy(y_or)
    decodedMsgs_irm = IRM(y_irm=y_2, K=K, beta=beta, H=H, N=N, V=V, prtDetail=False)
    accuracy_irm, EUIM_irm = evaluate_result(K=K, chosenNums=chosenNums, decodedMsgs=decodedMsgs_irm, rayleigh_scale=rayleigh_scale, beta=beta)
    _, distances_irm_toAdd= get_distanceInfo(K, chosenNums, decodedMsgs_irm)

    return [accuracy_vsic, EUIM_vsic, distances_vsic_toAdd, accuracy_irm, EUIM_irm, distances_irm_toAdd]


## Accept user inputs, specifying simulation arguments
parser = OptionParser()
parser.add_option("--args", type="string", dest="args", help="Arguments", default="")
parser.add_option("--K", type="int", dest="K", help="Number of users", default=-1)
parser.add_option("--N", type="int", dest="N", help="Number of channel use", default=-1)
parser.add_option("--M", type="int", dest="M", help="width of sensing matrix", default=-1)
parser.add_option("--RS", type="int", dest="RayleighScale", help="RayleighScale", default=-1)
parser.add_option("--noisePower", type="float", dest="noisePower", help="noisePower", default=-1)
parser.add_option("--num_trials", type="int", dest="num_trials", help="num of trials", default=-1)

(options, args) = parser.parse_args()

print( " --- --- --- --- --- ")

### Examine whether the user inputs are valid
K = options.K;                                  assert K > 0 
N = options.N;                                  assert N > 100
M = options.M;                                  assert M > N 
RayleighScale = options.RayleighScale;          assert RayleighScale > 0
noisePower = options.noisePower;                assert noisePower >= 0
num_trials = options.num_trials;                assert num_trials > 0

print("K, RayleighScale, noisePower, num_trials: ", K, RayleighScale, noisePower, num_trials)

V = 1

np.random.seed(8)

sum_acc_vsic = 0 
sum_acc_irm  = 0
sum_EUIM_vsic = 0
sum_EUIM_irm = 0
sum_dist_vsic = np.zeros(K, dtype=int)
sum_dist_irm  = np.zeros(K, dtype=int)

with tqdm_joblib(tqdm(desc="Progress", total=num_trials)) as progress_bar:
    a = Parallel(n_jobs=-1)( delayed(single_vsic_and_irm)(N, M, K, V, RayleighScale, noisePower) for _ in range(num_trials))

for j in range(num_trials):
    thisResult = a[j] 
    ## a[j] = [accuracy_vsic, EUIM_vsic, distances_vsic_toAdd, accuracy_irm, EUIM_irm, distances_irm_toAdd]
    sum_acc_vsic += thisResult[0] 
    sum_acc_irm  += thisResult[3]
    sum_EUIM_vsic+= thisResult[1]
    sum_EUIM_irm += thisResult[4]
    sum_dist_vsic+= thisResult[2]
    sum_dist_irm += thisResult[5]

print(" Acc of vsic: " + str(sum_acc_vsic / num_trials))
print(" Acc of IRM : " + str(sum_acc_irm  / num_trials))
print(" EUIM of vsic: " + str(sum_EUIM_vsic/ num_trials))
print(" EUIM of IRM : " + str(sum_EUIM_irm / num_trials))
print(" Prob. of mismatch index, vsic ", np.array(sum_dist_vsic) * (1/ sum(sum_dist_vsic))  )
print(" Prob. of mismatch index, IRM  ", np.array(sum_dist_irm ) * (1/ sum(sum_dist_irm ))  )