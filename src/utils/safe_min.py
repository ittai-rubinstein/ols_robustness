import numpy as np

def safe_min(arr):
    if arr.size == 0:
        return np.inf
    else:
        return np.min(arr)
