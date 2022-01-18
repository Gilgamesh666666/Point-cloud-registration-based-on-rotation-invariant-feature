import torch
import torch.nn as nn

import PVCNN.modules.functional as F

__all__ = ['Spherical_Voxelization']


class Spherical_Voxelization(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.r = int(resolution)
       
    def forward(self, features, coords):
        # 在外面做norm好了,这里就当做centriod在原点,且离原点最远的点已经norm为1
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        
        norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values + 1e-20)
        out, inds = F.spherical_avg_voxelize(features, norm_coords, self.r)
        
        inds = inds.detach()
        return out, inds, norm_coords

    def extra_repr(self):
        return 'resolution={}'.format(self.r)
