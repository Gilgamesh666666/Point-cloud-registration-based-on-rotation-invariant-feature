import torch.nn as nn
from PVCNN.modules.functional.knn import k_nearest_neighbor
import torch
class knnModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, k, bilateral, return_distance, return_index):
        # return [b, k, n] [b, k, m]
        dist1, dist2, idx1, idx2 = k_nearest_neighbor(input1, input2, k)
        if return_distance and return_index:
            if bilateral:
                return dist1.sqrt(), dist2.sqrt(), idx1, idx2
            else:
                return dist1.sqrt(), idx1
        elif return_distance and (not return_index):
            if bilateral:
                return dist1.sqrt(), dist2.sqrt()
            else:
                return dist1.sqrt()
        elif (not return_distance) and return_index:
            if bilateral:
                return idx1, idx2
            else:
                return idx1
        elif (not return_distance) and (not return_index):
            return