from torch.autograd import Function

from PVCNN.modules.functional.backend import _backend

__all__ = ['spherical_avg_voxelize']


class Spherical_AvgVoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution):
        """
        :param ctx:
        :param features: Features of the point cloud, FloatTensor[B, C, N]
        :param coords: Voxelized Coordinates of each point, IntTensor[B, 3, N]
        :param resolution: Voxel resolution
        :return:
            Voxelized Features, FloatTensor[B, C, R, R, R]
            voxel index of each point, IntTensor[B, N]
        """
        features = features.contiguous()
        coords = coords.contiguous()
        b, c, n = features.shape
        out, indices, counts = _backend.spherical_avg_voxelize_forward(features, coords, resolution)
        
        ctx.mark_non_differentiable(indices)
        ctx.save_for_backward(indices, counts)
        return out.view(b, c, resolution, resolution, resolution), indices.view(b, n)

    @staticmethod
    def backward(ctx, grad_output, tmp):
        """
        :param ctx:
        :param grad_output: gradient of output, FloatTensor[B, C, R, R, R]
        :return:
            gradient of inputs, FloatTensor[B, C, N]
        """
        b, c = grad_output.shape[:2]
        indices, counts = ctx.saved_tensors
        grad_features = _backend.spherical_avg_voxelize_backward(grad_output.contiguous().view(b, c, -1), indices, counts)
        return grad_features, None, None


spherical_avg_voxelize = Spherical_AvgVoxelization.apply
