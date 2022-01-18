import torch
import torch.nn as nn

from PVCNN.models.utils import create_pointnet_components, create_mlp_components
import PVCNN.modules.functional as F
from PVCNN.modules.shared_mlp import SharedMLP
from PVCNN.modules.ball_query import BallQuery
__all__ = ['shapenet_PVCNN']


class shapenet_PVCNN(nn.Module):
    def __init__(self, blocks, point_kernel_formal, voxel_shape, num_classes, num_shapes,
    extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1, 
    rot_invariant_preprocess=None, with_local_feat=False):
        super().__init__()
        assert extra_feature_channels >= 0
        self.in_channels = extra_feature_channels + 3
        self.num_shapes = num_shapes
        self.rot_invariant_preprocess = rot_invariant_preprocess
        self.with_local_feat = with_local_feat
        if rot_invariant_preprocess=='ppf':
            print('use ppf')
            assert extra_feature_channels >= 3
            self.in_channels = 4
        elif rot_invariant_preprocess=='new_ppf':
            print('use new ppf')
            assert extra_feature_channels >= 3
            self.in_channels = 5
        elif rot_invariant_preprocess=='change_coords':
            print('use change_coords')
            self.in_channels = extra_feature_channels + 3
        elif rot_invariant_preprocess=='pca':
            print('use pca')
            self.in_channels = extra_feature_channels + 3
            self.FourClassify_dim = 128
            layers1, _ = create_mlp_components(in_channels=self.in_channels, 
                                            out_channels=[64, 0.2, 128, 0.2, self.FourClassify_dim],
                                            classifier=False, dim=2, width_multiplier=width_multiplier)
            layers2, _ = create_mlp_components(in_channels=self.FourClassify_dim, 
                                            out_channels=[64, 4],
                                            classifier=True, dim=1, width_multiplier=width_multiplier)
            self.FourFeature = nn.Sequential(*layers1)
            self.FourClassify = nn.Sequential(*layers2)
        elif rot_invariant_preprocess is None:
            print('with no need for rotation invariance')
            self.in_channels = extra_feature_channels + 3
        
        if with_local_feat:
            self.radius = 0.3
            self.neighbor_num = 128
            self.grouper = BallQuery(self.radius, self.neighbor_num, include_coordinates=True)
            self.ppf_fuse_dim = 64
            self.fuser = SharedMLP(4, [32, self.ppf_fuse_dim], dim=2)
            self.in_channels += self.ppf_fuse_dim
        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=blocks, point_kernel_formal=point_kernel_formal, voxel_shape=voxel_shape, 
            in_channels=self.in_channels, with_se=False, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.point_features = nn.ModuleList(layers)

        layers, _ = create_mlp_components(in_channels=(num_shapes + channels_point + concat_channels_point),
                                          out_channels=[256, 0.2, 256, 0.2, 128, num_classes],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        # inputs : [B, in_channels + S, N]
        one_hot_vectors = inputs[:, -self.num_shapes:, :]
        
        coords = inputs[:, :3, :]
        
        b, _, n = coords.shape
        if self.rot_invariant_preprocess=='ppf':
            normals = inputs[:, 3:6, :]
            normals = normals/normals.norm(dim=1, keepdim=True)
            
            centers_coords = coords.mean(dim=2, keepdim=True).expand(-1, -1, n)
            centers_normals = normals.mean(dim=2, keepdim=True).expand(-1, -1, n)
            global_ppfs = F.ppf(centers_coords, coords, centers_normals, normals) # [b, 4, n]
            features = global_ppfs
            
        elif self.rot_invariant_preprocess=='new_ppf':
            normals = inputs[:, 3:6, :]
            normals = normals/normals.norm(dim=1, keepdim=True)
            
            centers_coords = coords.mean(dim=2, keepdim=True) # [b, 3, 1]
            centers_normals = normals.mean(dim=2, keepdim=True) # [b, 3, 1]
            norm_centers_normals = centers_normals/centers_normals.norm(dim=1, keepdim=True) # [b, 3, 1]
            old_ppfs = F.ppf(centers_coords.expand(-1, -1, n), coords, centers_normals.expand(-1, -1, n), normals) # [b, 4, n]
            norm_coords = coords - centers_coords # [b, 3, n]
            # [b, 3, n]->[b, n, 3]->[b, n, 1]->[b, 1, n] * [b, 3, n] = [b, 3, n]
            project_coords = norm_coords - (norm_coords.permute(0, 2, 1).bmm(norm_centers_normals).permute(0, 2, 1))*(centers_normals.expand(-1, -1, n))
            cos_alpha = project_coords.permute(0, 2, 1).bmm(project_coords) # [b, n, n]
            sin_alpha = project_coords.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, n, -1).cross(project_coords.permute(0, 2, 1).unsqueeze(1).expand(-1, n, -1, -1)).norm(dim=3)
            atan_alpha = torch.atan2(sin_alpha, cos_alpha) # [b, n, n]
            atan_alpha[atan_alpha<=1e-5] = 100 # [b, n, n]
            alpha = atan_alpha.median(dim=2, keepdim=True).values # [b, n, 1]
            global_new_ppfs = torch.cat((old_ppfs, alpha.permute(0, 2, 1)), dim=1) # [b, 5, n]
            
            # # [b, 3, k, m=n]->[b, m=n, k, 3].matmul [b, m=n, 3, 1]->[b, m=n, k, 1]->[b, 3, k, m=n] * [b, 3, k, m=n] = [b, 3, k, m=n]
            # project_coords = d - (d.permute(0, 3, 2, 1).matmul(center_normals.unsqueeze(2).permute(0, 3, 1, 2)).permute(0, 3, 2, 1))*center_normals_k_repeat
            # cos_alpha = project_coords.permute(0, 3, 2, 1).matmul(project_coords.permute(0, 3, 1, 2)) # [b, m=n, k, k]
            # sin_alpha = project_coords.permute(0, 3, 2, 1).unsqueeze(3).expand(-1, -1, -1, self.neighbor_num, -1).cross(project_coords.permute(0, 3, 2, 1).unsqueeze(2).expand(-1, -1, self.neighbor_num, -1, -1)).norm(dim=4)
            # atan_alpha = torch.atan2(sin_alpha, cos_alpha) # [b, m=n, k, k]
            # atan_alpha[atan_alpha<=1e-5] = 100 # [b, m=n, k, k]
            # # [b, m=n, k, k]->[b, k, k, m=n]
            # alpha = atan_alpha.permute(0, 2, 3, 1).median(dim=2, keepdim=True).values # [b, k, 1, m=n]

            # local_new_ppfs = torch.cat((local_ppf, alpha.permute(0, 2, 1, 3)), dim=1) # (b, 5, k, m=n)
            # local_features = self.fuser(local_new_ppfs).max(dim=2).values # [b, c, m=n]
            features = global_new_ppfs # [b, 5, m=n]
            #print(f'centers_coords = {centers_coords.permute(0, 2, 1)}')
            # 加一层保险吧
            #features[mask.unsqueeze(1).expand(-1, 4, -1)] = 0
        elif self.rot_invariant_preprocess=='change_coords':
            norm_coords = coords - coords.mean(dim=2, keepdim=True) # [b, 3, n]
            rank = torch.argsort(norm_coords.norm(dim=1), dim=1, descending=True) #[b, n]
            batch_base_x = torch.zeros(b, 3 ,1).to(norm_coords) # [b, 3, 1]
            batch_base_y = torch.zeros(b, 3 ,1).to(norm_coords) # [b, 3, 1]
            for i in range(b):
                base_x = norm_coords[i, :, rank[i, 0]] # [3,]
                assert (base_x.norm() > 1e-5)
                base_x = base_x/base_x.norm()# [3,]
                for j in range(1, n):
                    base_y = norm_coords[i, :, rank[i, j]] # [3,]
                    if base_y.norm() < 1e-5:
                        continue
                    base_y = base_y/base_y.norm() # [3,]
                    lamda = (base_x * base_y).sum()
                    if(lamda < 0.9 and lamda > -0.9):
                        break
                try:
                    assert (lamda < 0.9 and lamda > -0.9)
                except:
                    print(j)
                    print(lamda)
                batch_base_x[i, :, :] = base_x.unsqueeze(1)
                batch_base_y[i, :, :] = base_y.unsqueeze(1)
            
            # orthogonality
            batch_base_x -= batch_base_y*(batch_base_x.permute(0, 2, 1).bmm(batch_base_y)) # [b, 1, 1]
            assert (batch_base_x.norm(dim=1, keepdim=True) < 1e-5).sum()<1
            batch_base_x /= batch_base_x.norm(dim=1, keepdim=True)
            # 别忘了norm
            batch_base_z = batch_base_x.cross(batch_base_y, dim=1) # [b, 3, 1]
            batch_base_z = batch_base_z/batch_base_z.norm(dim=1, keepdim=True)
            new_x = batch_base_x.permute(0, 2, 1).bmm(norm_coords) # [b, 1, n]
            new_y = batch_base_y.permute(0, 2, 1).bmm(norm_coords) # [b, 1, n]
            new_z = batch_base_z.permute(0, 2, 1).bmm(norm_coords)  # [b, 1, n]
            features = torch.cat((new_x, new_y, new_z), dim=1)
            #coords = features
            #print(f'features = {features}')
            if self.in_channels >= 6:
                normals = inputs[:, 3:6, :]
                centers_coords = coords.mean(dim=2, keepdim=True).expand(-1, -1, n)
                centers_normals = normals.mean(dim=2, keepdim=True).expand(-1, -1, n)
                ppfs = F.ppf(centers_coords, coords, centers_normals, normals)
                # new_nx = batch_base_x.permute(0, 2, 1).bmm(normals) # [b, 1, n]
                # new_ny = batch_base_y.permute(0, 2, 1).bmm(normals) # [b, 1, n]
                # new_nz = batch_base_z.permute(0, 2, 1).bmm(normals)  # [b, 1, n]
                #features = torch.cat((new_x, new_y, new_z, new_nx, new_ny, new_nz), dim=1)
                features = torch.cat((new_x, new_y, new_z, ppfs), dim=1)
            
        elif self.rot_invariant_preprocess == 'pca':
            # x [b, 3, n]
            # source pca
            s_ba = torch.mean(coords, axis=2, keepdim=True) # [b, 3, 1]
            s = coords - s_ba  # [b, 3, n]
            su, _, _ = torch.svd(s) # [b, 3, 3]

            FourClassifyscore = self.FourClassify(self.FourFeature(s).max(dim=2).values)
            #return FourClassifyscore
            reflect = FourClassifyscore.softmax(dim=1).max(dim=1).indices # [b,]
            a = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]]).to(s) # [4, 2]
            a = a.unsqueeze(0).expand(b, -1, -1) # [b, 4, 2]
            #print(a.shape)
            b = a.gather(1, reflect.unsqueeze(1).unsqueeze(2).expand(-1, -1, 2)) # [b, 1, 2]
            sb32 = su[:, :, :2] * b
            sb31 = sb32[:, :, 0].cross(sb32[:, :, 1]).unsqueeze(2)
            sb33 = torch.cat((sb32, sb31), dim=2) # [b, 3, 3]
            after_s = sb33.permute(0,2,1).bmm(s) # [b, 3, n]
            features = after_s
            coords = after_s

            # a = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]]).to(s) # [4, 2]
            # a = a.unsqueeze(0).unsqueeze(2).expand(b, -1, -1, -1) # [b, 4, 1, 2]
            
            # sb432 = su[:, :, :2].unsqueeze(1).expand(-1, 4, -1, -1) * a
            # sb431 = sb432[:, :, :, 0].cross(sb432[:, :, :, 1]).unsqueeze(3)
            # sb433 = torch.cat((sb432, sb431), dim=3) # [b, 4, 3, 3]
            # #print(s.unsqueeze(1).expand(-1, 4, -1, -1).contiguous().view(-1, 3, n).shape)
            # after_s = sb433.contiguous().view(-1, 3, 3).permute(0,2,1).bmm(s.unsqueeze(1).expand(-1, 4, -1, -1).contiguous().view(-1, 3, n)) # [b*4, 3, n]
            # #print(u.shape)
            # if self.training:
            #     features = after_s
            #     #coords = coords.unsqueeze(1).expand(-1, 4, -1, -1).contiguous().view(-1, 3, n)
            #     coords = after_s
            # else:
            #     features = su.permute(0,2,1).bmm(s)
        elif self.rot_invariant_preprocess is None:
            features = inputs
        #print(inputs.shape)
        if self.with_local_feat:
            # 已经除去center自己
            center_coords = coords
            center_normals = normals
            neighbor_coords_normals = self.grouper(coords, center_coords, normals) # [b, 6, k, m=n]
            neighbor_coords = neighbor_coords_normals[:, :3, :, :] # [b, 3, k, m=n]
            neighbor_normals = neighbor_coords_normals[:, 3:, :, :] # [b, 3, k, m=n]
            center_coords_k_repeat = center_coords.unsqueeze(2).expand(-1, -1, self.neighbor_num, -1) # (b, 3, k, m=n)
            center_normals_k_repeat = center_normals.unsqueeze(2).expand(-1, -1, self.neighbor_num, -1) # (b, 3, k, m=n)
            d = center_coords_k_repeat - neighbor_coords # (b, 3, k, m=n)
            d_norm = torch.norm(d, dim=1, p=2, keepdim=True) # (b, 1, k, m=n)
            d_unit = d/d_norm # (b, 3, k, m)
            nr_d = torch.acos(neighbor_normals.mul(d_unit).sum(dim=1, keepdim=True).clamp(-1, 1)) # (b, 1, k, m=n)
            ni_d = torch.acos(center_normals_k_repeat.mul(d_unit).sum(dim=1, keepdim=True).clamp(-1, 1)) # (b, 1, k, m=n)
            nr_ni = torch.acos(neighbor_normals.mul(center_normals_k_repeat).sum(dim=1, keepdim=True).clamp(-1, 1)) # (b, 1, k, m=n)
            local_ppf = torch.cat((nr_d, ni_d, nr_ni, d_norm), dim=1) # (b, 4, k, m=n)
            local_features = self.fuser(local_ppf).max(dim=2).values # (b, c, m=n)
            features = torch.cat((features, local_features), dim=1)
        out_features_list = [one_hot_vectors]
        for i in range(len(self.point_features)):
            features, _ = self.point_features[i]((features, coords))
            out_features_list.append(features)
        out_features_list.append(features.max(dim=-1, keepdim=True).values.repeat([1, 1, n]))
        return self.classifier(torch.cat(out_features_list, dim=1))
