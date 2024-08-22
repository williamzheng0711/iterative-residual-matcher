import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import copy
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
# from parallel import ParallelTqdm


class ParallelTqdm(Parallel):
    """joblib.Parallel, but with a tqdm progressbar
    Additional parameters:
    ----------------------
    total_tasks: int, default: None
        the number of expected jobs. Used in the tqdm progressbar.
        If None, try to infer from the length of the called iterator, and
        fallback to use the number of remaining items as soon as we finish
        dispatching.
        Note: use a list instead of an iterator if you want the total_tasks
        to be inferred from its length.
    desc: str, default: None
        the description used in the tqdm progressbar.
    disable_progressbar: bool, default: False
        If True, a tqdm progressbar is not used.
    show_joblib_header: bool, default: False
        If True, show joblib header before the progressbar.
    Removed parameters:
    -------------------
    verbose: will be ignored
    Usage:
    ------
    >>> from joblib import delayed
    >>> from time import sleep
    >>> ParallelTqdm(n_jobs=-1)([delayed(sleep)(.1) for _ in range(10)])
    80%|████████  | 8/10 [00:02<00:00,  3.12tasks/s]
    """

    def __init__(
        self,
        *,
        total_tasks: int | None = None,
        desc: str | None = None,
        disable_progressbar: bool = False,
        show_joblib_header: bool = False,
        **kwargs
    ):
        if "verbose" in kwargs:
            raise ValueError(
                "verbose is not supported. "
                "Use show_progressbar and show_joblib_header instead."
            )
        super().__init__(verbose=(1 if show_joblib_header else 0), **kwargs)
        self.total_tasks = total_tasks
        self.desc = desc
        self.disable_progressbar = disable_progressbar
        self.progress_bar: tqdm.tqdm | None = None

    def __call__(self, iterable):
        try:
            if self.total_tasks is None:
                # try to infer total_tasks from the length of the called iterator
                try:
                    self.total_tasks = len(iterable)
                except (TypeError, AttributeError):
                    pass
            # call parent function
            return super().__call__(iterable)
        finally:
            # close tqdm progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()

    __call__.__doc__ = Parallel.__call__.__doc__

    def dispatch_one_batch(self, iterator):
        # start progress_bar, if not started yet.
        if self.progress_bar is None:
            self.progress_bar = tqdm(
                desc=self.desc,
                total=self.total_tasks,
                disable=self.disable_progressbar,
                unit="tasks",
            )
        # call parent function
        return super().dispatch_one_batch(iterator)

    dispatch_one_batch.__doc__ = Parallel.dispatch_one_batch.__doc__

    def print_progress(self):
        """Display the process of the parallel execution using tqdm"""
        # if we finish dispatching, find total_tasks from the number of remaining items
        if self.total_tasks is None and self._original_iterator is None:
            self.total_tasks = self.n_dispatched_tasks
            self.progress_bar.total = self.total_tasks
            self.progress_bar.refresh()
        # update progressbar
        self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)

def lauch_once_MLSIC(m,N,K,V, rayleigh_scale):
    H = np.random.normal(scale=1/np.sqrt(m), size=(m, N))
    beta = np.random.rayleigh(scale=rayleigh_scale, size=K)
    # beta = 5 * np.ones(shape=(K))
    # sort the vector from largest to smallest
    beta = np.sort(beta)[::-1]
    # print("These are the channels: ", beta, beta[0]**2, np.linalg.norm(beta[1:],2)**2 )

    # generating the received signal y
    # z = np.random.normal(scale=1, size=m)
    z = 0
    # chosenNums = np.random.choice(N, K, replace=False)
    # chosenNums = np.sort(chosenNums)
    chosenNums = np.arange(K)
    # print("These are the index of chosen cdwds: ", chosenNums)
    y_ml = np.sqrt(V) * H[:, chosenNums] @ beta + z
    y_ml = np.expand_dims(y_ml, axis=-1)

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

    # toAdd = [decodedMsg in chosenNums for decodedMsg in decodedMsgsML]

    distances = np.array([ np.abs(decodedMsg - chosenNums[idx]) if decodedMsg in chosenNums else 10*K for idx, decodedMsg in enumerate(decodedMsgsML)], dtype=int)

    return [ np.count_nonzero(distances == n) for n in range(K) ]


# Define your matrix dimensions
m, N = 700, 200000
K = 100
V = 1
rayleigh_scale = 5

num_trial = 200

correct_freq_1 = np.zeros(K, dtype=int)
args = [[m, N, K, V, rayleigh_scale]  for j in range(num_trial)]

for j in tqdm(range(num_trial)):
    temp = lauch_once_MLSIC(m,N,K,V,rayleigh_scale)
    correct_freq_1 = correct_freq_1 + np.array(temp)


# array_needed = np.array(all_results).reshape((num_trial, K))
# correct_freq_1 = np.sum(array_needed, axis=0)

plt.plot(range(K), 1/num_trial * correct_freq_1)
plt.show()

print(num_trial, correct_freq_1)