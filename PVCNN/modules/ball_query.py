import torch
import torch.nn as nn

import PVCNN.modules.functional as F

__all__ = ['BallQuery']


class BallQuery(nn.Module):
    def __init__(self, radius, num_neighbors, include_coordinates=True):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.include_coordinates = include_coordinates

    def forward(self, points_coords, centers_coords, points_features=None):
        # [b, 3, n]
        points_coords = points_coords.contiguous()
        centers_coords = centers_coords.contiguous()
        # [b, m, u]
        neighbor_indices = F.ball_query(centers_coords, points_coords, self.radius, self.num_neighbors)
        # [b, 3, m, u]
        neighbor_coordinates = F.grouping(points_coords, neighbor_indices)
        neighbor_coordinates = neighbor_coordinates - centers_coords.unsqueeze(-1)
        # [b, c, n]
        if points_features is None:
            assert self.include_coordinates, 'No Features For Grouping'
            neighbor_features = neighbor_coordinates
        else:
            # [b, c, m, u]
            neighbor_features = F.grouping(points_features, neighbor_indices)
            if self.include_coordinates:
                # [b, c + 3, m, u]
                neighbor_features = torch.cat([neighbor_coordinates, neighbor_features], dim=1)
        return neighbor_features.permute(0, 1, 3, 2) #[b, c, m, u]->[b, c, u, m]

    def extra_repr(self):
        return 'radius={}, num_neighbors={}{}'.format(
            self.radius, self.num_neighbors, ', include coordinates' if self.include_coordinates else '')
