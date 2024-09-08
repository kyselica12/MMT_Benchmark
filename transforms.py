import numpy as np
import torch

def allworth_transform(X):
    input_size = 1_200
    for ch in ["mag", "phase"]:
        x = X[ch] 
        x = x[:, :input_size*8].reshape(1, input_size, -1)
        count = (x != 0).sum(axis=2)
        count[count == 0] = 1 # avoid division by zero
        sum_ = x.sum(axis=2)
        X[ch] = sum_ / count
    return X

def furfaro_transform(X):
    X["mag"] = X["mag"][:, :500]
    return X

def astroconformer_transform(X):
    X["mag"] = X["mag"].reshape(-1)
    return X

def yao_transform(X):
    input_size = 200
    p = X["period"].item()
    x = X["mag"]
    y = np.zeros((1, 200))
    c = np.zeros((1, input_size))
    step = p / input_size

    for i, val in enumerate(x[0]):
        idx = int(round((i-step/2) / step)) % input_size
        y[0][idx] += val
        c[0][idx] += 1
    
    y[y!=0] = y[y!=0] / c[y!=0]

    X["mag"] = y

    return X