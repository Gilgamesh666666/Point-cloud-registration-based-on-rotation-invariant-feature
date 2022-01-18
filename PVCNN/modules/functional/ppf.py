from torch.autograd import Function

from PVCNN.modules.functional.backend import _backend

__all__ = ['ppf']


def ppf(centers_coords, points_coords, centers_normals, points_normals):
        """
        :param centers_coords: coordinates of centers, FloatTensor[B, 3, N]
        :param points_coords: coordinates of points, FloatTensor[B, 3, N]
        :param centers_normals: normals of centers, FloatTensor[B, 3, N]
        :param points_normals: normals of points, FloatTensor[B, 3, N]
        :return:
            feat: features, FloatTensor[b, 4, n]
        """
        #print(centers_coords)
        centers_coords = centers_coords.contiguous()
        points_coords = points_coords.contiguous()
        centers_normals = centers_normals.contiguous()
        points_normals = points_normals.contiguous()
        return _backend.spherical_ppf_forward(points_coords, centers_coords, points_normals, centers_normals)
