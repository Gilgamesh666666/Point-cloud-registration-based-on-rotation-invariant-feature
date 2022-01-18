import torch
import torch.nn as nn

import PVCNN.modules.functional as F

__all__ = ['Voxelization']


class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        # 把coords变成vox_coords
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        # 不管之前怎么norm怎么弄,voxelization的过程就是
        # 把coords限制(0, self.r - 1)内然后在简单的向下取整
        # 所以devoxelization里面只要默认这种方式进行反推就可以了
        vox_coords = torch.round(norm_coords).to(torch.int32)
        #print(f'features = {features}')
        out, indices = F.avg_voxelize(features, vox_coords, self.r)
        #print(f'out = {out}')
        #print(vox_coords.requires_grad)
        indices = indices.detach()
        return out, indices, norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')
