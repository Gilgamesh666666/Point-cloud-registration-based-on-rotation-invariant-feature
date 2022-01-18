'''
Author: your name
Date: 2020-11-20 22:28:32
LastEditTime: 2020-11-24 22:35:59
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /exp1/PVCNN/models/pointnet_classify.py
'''
# 用来测试pca的
import torch.nn as nn
import os
class ShareMlp(nn.Module):
    def __init__(self, inchannel, outchannels, dim=1):
        super().__init__()
        # [b, c, n]/[b, c]
        ic = inchannel
        layers = []
        if dim == 1:
            conv = nn.Linear
        else:
            conv = nn.Conv1d
        
        for oc in outchannels:
            layers.extend([
                conv(ic, oc, 1),
                nn.BatchNorm1d(oc),
                nn.ReLU(True)])
            ic = oc
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class  Classifer(nn.Module):
    def __init__(self, inchannel, outchannels, scale, dim=1):
        super().__init__()
        # [b, c, n]/[b, c]
        ic = inchannel
        layers = []
        for oc in outchannels:
            if oc < 1:
                layers.append(nn.Dropout(oc))
            else:
                oc = int(oc/scale)
                layers.append(ShareMlp(ic, [oc], dim=dim)) # [b, c', n]
                ic = oc
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

def pca(x):
    # x [b, 3, n]
    norm_x = x - torch.mean(x, axis=2, keepdim=True)
    u, _, _ = torch.svd(norm_x) # [b, 3, 3]
    #print(u.shape)
    return u.permute(0, 2, 1).bmm(norm_x)

class pointnet(nn.Module):
    def __init__(self, inchannel, outchannels, num_class, rot_invariant=None):
        super().__init__()
        self.mlps = ShareMlp(inchannel, outchannels, dim=2)
        self.classifer = Classifer(outchannels[-1], [512, 0.2, 256, num_class], 1, dim=1)
        self.rot_invariant = rot_invariant
    def forward(self, x):
        # [b, 3, n]
        # if self.rot_invariant is not None:
        #     x = pca(x)
        return self.classifer(self.mlps(x).max(2).values)
