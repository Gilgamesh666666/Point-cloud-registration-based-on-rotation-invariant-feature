import numpy as np
import torch
# 引用FCGF
def _hash(arr, seed):
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, d] * seed**d
        else:
            hash_vec += arr[d] * seed**d
    return hash_vec
# https://www.zhihu.com/question/40920696/answer/88858919。
def find_row(row, mat:np.array):
    # [m,], [n, m]
    return np.where((row==mat).all(1))[0]

def filter_intersection(source_mat:np.array, set_mat:np.array):
    # [n1,m], [n2, m]
    n1 = source_mat.shape[0]
    mask = np.zeros(n1,dtype=bool)
    for i in range(n1):
        if len(find_row(source_mat[i, :], set_mat)):
            mask[i] = True
    return source_mat[np.logical_not(mask),:]

def get_hash_key_for_pairs(idx1:torch.tensor, idx2:torch.tensor, seed:int):
    # [n,], [n,]
    return _hash(torch.cat((idx1.unsqueeze(1), idx2.unsqueeze(1)), axis=1), seed)