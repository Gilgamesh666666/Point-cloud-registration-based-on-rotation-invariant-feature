import torch
import torch.nn as nn

import PVCNN.modules.functional as F
from .voxelization import Voxelization
from .spherical_vox import Spherical_Voxelization
from .shared_mlp import SharedMLP
from .se import SE3d
import copy

import numpy as np
__all__ = ['PVConv']


class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, point_kernel_formal, voxel_shape, kernel_size, resolution, with_coeff=False, with_se=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.point_kernel_formal = point_kernel_formal
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.voxel_shape = voxel_shape
        self.with_coeff = with_coeff
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        self.spherical_vox = Spherical_Voxelization(resolution)
        if self.with_coeff:
            self.coefficient = nn.Parameter(torch.Tensor([1]))
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        if self.point_kernel_formal == 'dgcnn_kernel':
            in_channels *= 2
        self.point_layers = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        b, c, n = features.shape
        if self.voxel_shape=='cube':
            #print('use cube')
            avg_voxel_features, inds, voxel_coords = self.voxelization(features, coords)
            #print(inds)
            voxel_features = self.voxel_layers(avg_voxel_features)
            voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        elif self.voxel_shape=='spherical':
            #print('use spherical')
            # ????????????undefined point?????????????????????,?????????avg_voxel_features???
            # ?????????????????????????????????,???????????????????????????????????????voxel?????????feature???
            # pointnet?????????,???????????????feature??????????????????,???????????????feature
            # ?????????????????????????????????,????????????????????????????????????feature,
            # ?????????feature??????????????????,??????????????????????????????
            # ????????????,?????????pvconv???,??????????????????feature??????????????????????????????
            # ?????????????????????????????????coords???voxel,
            # ???????????????voxel?????????inds???????????????,?????????????????????????????????,
            # ????????????pvconv???,?????????????????????feature????????????????????????feature??????
            avg_voxel_features, inds, voxel_coords =  self.spherical_vox(features, coords)            
            assert avg_voxel_features.is_contiguous()
            voxel_features = self.voxel_layers(avg_voxel_features)
            voxel_features = F.spherical_trilinear_devoxelize(voxel_features, voxel_coords, inds, self.resolution, self.training)
        if self.point_kernel_formal == 'dgcnn_kernel':
            #print('use dgcnn_kernel')
            avg_voxel_features = avg_voxel_features.view(b, c, -1)#(b, c, r*r*r)
            # (b, c, n)
            # handle undefined points
            mask = (inds==-1) #[b, n]
            inds_temp = copy.deepcopy(inds)
            inds_temp[mask] = 0 # ??????????????????,??????mask??????,???????????????????????????-1,??????????????????
            # [b, c, n]
            #np.savetxt('inds_temp', inds_temp.cpu().numpy(), delimiter=',')
            upbound = avg_voxel_features.shape[2]-1
            lowbound = 0
            inds_temp = inds_temp.unsqueeze(1).expand(-1, c, -1).long()
            if((inds_temp.gt(upbound).sum()>0) or (inds_temp.lt(lowbound).sum()>0)):
                print(inds_temp.gt(upbound).sum(), inds_temp.lt(lowbound).sum())
                print(f'upbound = {upbound}\noverflow = {inds_temp[inds_temp.gt(upbound)]}')
                print(f'underflow = {inds_temp[inds_temp.lt(lowbound)]}')
            center_features = avg_voxel_features.gather(2, inds_temp)
            related_features = features - center_features
            # if center point is a undefined_point, 
            # then the related_features of point are set to zeros
            related_features[mask.unsqueeze(1).expand(-1, c, -1)] = 0
            point_features = self.point_layers(torch.cat((related_features, features), 1))
        elif self.point_kernel_formal == 'pointnet_kernel':
            #print('use pointnet_kernel')
            point_features = self.point_layers(features)
        if self.with_coeff:
            fused_features = self.coefficient*voxel_features + point_features
            #print(self.coefficient)
        else:
            fused_features = voxel_features + point_features
        return fused_features, coords

    