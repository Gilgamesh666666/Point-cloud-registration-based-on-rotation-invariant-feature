# functions/add.py
import torch
from PVCNN.modules.functional.backend import _backend
__all__ = ['nearest_neighbor']

class K_Nearest_Neighbor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, k):
        # default that xyz1 = [b, c, n]
        xyz1 = xyz1.float().contiguous()
        xyz2 = xyz2.float().contiguous()
        dist1, dist2, idx1, idx2 = _backend.knn_forward_cuda(xyz1, xyz2, k)
        ctx.mark_non_differentiable(idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1,idx2)
        return dist1, dist2, idx1, idx2
    @staticmethod
    def backward(ctx, graddist1, graddist2, temp1, temp2):
        #print(self.idx1, self.idx2)
        xyz1,xyz2,idx1,idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        
        gradxyz1, gradxyz2 = _backend.knn_backward_cuda(xyz1, xyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2, None
k_nearest_neighbor = K_Nearest_Neighbor.apply