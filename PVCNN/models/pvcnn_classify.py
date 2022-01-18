import open3d as o3d
import numpy as np
import torch
import torch.nn as nn

from .utils import create_mlp_components, create_pointnet_components
import torch.nn.functional as torchF
import PVCNN.modules.functional as F
from PVCNN.modules.spherical_vox import Spherical_Voxelization
from PVCNN.modules.ball_query import BallQuery
from PVCNN.modules.shared_mlp import SharedMLP
__all__ = ['PVCNN_classifier']

class PVCNN_classifier(nn.Module):
    
    def __init__(self, blocks, dim_k, point_kernel_formal, voxel_shape, num_classes,
    with_coeff=False, with_se=True, extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1, 
    is_classify=True, rot_invariant_preprocess=None, with_local_feat=None, 
    with_transform_fine_tune=False, use_new_coords_for_voxel=True):
        super().__init__()
        #assert with_coeff
        assert extra_feature_channels >= 0
        self.extra_feature_channels = extra_feature_channels
        self.is_classify = is_classify
        self.rot_invariant_preprocess = rot_invariant_preprocess
        self.with_local_feat = with_local_feat
        self.with_transform_fine_tune = with_transform_fine_tune
        self.use_new_coords_for_voxel = use_new_coords_for_voxel
        if rot_invariant_preprocess=='ppf':
            print('use ppf')
            #assert isinstance(ppf_vox_re, int)
            assert extra_feature_channels >= 3
            #self.ppf_spherical_vox = Spherical_Voxelization(ppf_vox_re)
            self.in_channels = 4
        elif rot_invariant_preprocess=='new_ppf':
            print('use new ppf')
            #assert isinstance(ppf_vox_re, int)
            assert extra_feature_channels >= 3
            #self.ppf_spherical_vox = Spherical_Voxelization(ppf_vox_re)
            self.in_channels = 5
        elif rot_invariant_preprocess=='change_coords':
            print('use change_coords')
            self.in_channels = extra_feature_channels + 3
        elif rot_invariant_preprocess=='pca':
            print('use pca')
            self.in_channels = extra_feature_channels + 3
            # self.FourClassify_dim = 128
            # layers1, _ = create_mlp_components(in_channels=self.in_channels, 
            #                                 out_channels=[64, 0.2, 128, 0.2, self.FourClassify_dim],
            #                                 classifier=False, dim=2, width_multiplier=width_multiplier)
            # layers2, _ = create_mlp_components(in_channels=self.FourClassify_dim, 
            #                                 out_channels=[64, 4],
            #                                 classifier=True, dim=1, width_multiplier=width_multiplier)
            # self.FourFeature = nn.Sequential(*layers1)
            # self.FourClassify = nn.Sequential(*layers2)
        elif rot_invariant_preprocess is None:
            print('with no need for rotation invariance')
            self.in_channels = extra_feature_channels + 3
        
        if with_local_feat is not None:
            self.radius = 0.3
            self.neighbor_num = 128
            self.grouper = BallQuery(self.radius, self.neighbor_num, include_coordinates=True)
            self.fuse_dim = 64
            if with_local_feat=='ppf':
                local_in_channel=4
                self.fuser = SharedMLP(local_in_channel, [32, self.fuse_dim], dim=2)
            elif with_local_feat=='change_coords':
                local_in_channel=3
            elif with_local_feat=='fpfh':
                local_in_channel=33
                #self.fuse_dim = 33
                self.fuser = SharedMLP(local_in_channel, [self.fuse_dim, self.fuse_dim], dim=1)
            self.in_channels += self.fuse_dim
        
        if with_transform_fine_tune:
            transfrom_dim = 32
            self.extract_feature_for_transform_block = SharedMLP(3, [32, transfrom_dim], dim=1)
            self.transform_block = nn.Sequential(*(create_mlp_components(transfrom_dim, [int(transfrom_dim/2), 6], classifier=True, dim=1, width_multiplier=1)[0]))

        #print(self.neighbor_num)
        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=blocks, point_kernel_formal=point_kernel_formal, voxel_shape=voxel_shape, 
            in_channels=self.in_channels, with_coeff=with_coeff, with_se=with_se, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.point_features = nn.ModuleList(layers)
        #if self.is_classify:
        layers, _ = create_mlp_components(in_channels=dim_k, 
                                        out_channels=[512, 0.2, 256, num_classes],
                                        classifier=True, dim=1, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        # inputs : [B, in_channels, N]
        b, _, n = inputs.shape
        coords = inputs[:, :3, :] # [B, 3, n]
        coords = coords - coords.mean(dim=2, keepdim=True) # [b, 3, n]
        if self.rot_invariant_preprocess=='ppf':
            normals = inputs[:, 3:6, :]
            normals = normals/normals.norm(dim=1, keepdim=True)
            # 用voxel做ppf
            # coords_normals = inputs[:, :6, :] # [b, 3, n]
            # voxel_centers_coords_normals, inds, voxel_coords = self.ppf_spherical_vox(coords_normals, coords)
            # voxel_centers_coords_normals = voxel_centers_coords_normals.contiguous().view(b, 6, -1)
            # # handle undefined points
            # mask = (inds==-1) #[b, n]
            # inds[mask] = voxel_centers_coords_normals.shape[2]
            # voxel_centers_coords_normals = torch.cat((voxel_centers_coords_normals, torch.zeros(b, 6, 1).to(inputs)), 2)
            # #
            # centers_coords_normals = voxel_centers_coords_normals.gather(2, inds.unsqueeze(1).expand(-1, 6, -1).long())
            # centers_coords = centers_coords_normals[:, :3, :]
            # centers_normals = centers_coords_normals[:, 3:6, :]
            centers_coords = coords.mean(dim=2, keepdim=True).expand(-1, -1, n)
            centers_normals = normals.mean(dim=2, keepdim=True).expand(-1, -1, n)
            global_ppfs = F.ppf(centers_coords, coords, centers_normals, normals) # [b, 4, n]
            features = global_ppfs
            
            # 加一层保险吧
            #features[mask.unsqueeze(1).expand(-1, 4, -1)] = 0
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
                assert (lamda < 0.9 and lamda > -0.9)
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
            # 放进change_coords里再试一次
            if self.with_transform_fine_tune:
                # assume that the coords have been normalized
                R6 = self.transform_block(self.extract_feature_for_transform_block(coords).max(dim=2).values)#[b, 6]
                R32 = torchF.normalize(R6.unsqueeze(2).view(-1, 2, 3), dim=2)# [b, 2, 3]
                a1, a2 = R32[:, 0, :], R32[:, 1, :] # [b, 3], [b, 3]
                b1 = a1 # [b, 3]
                b2 = a2 - ((a2 * b1).sum(dim=1, keepdim=True))*b1 # [b, 3]
                b2 = torchF.normalize(b2, dim=1) # [b, 3]
                b3 = torch.cross(b1, b2) # [b, 3]
                R = torch.cat((b1.unsqueeze(2), b2.unsqueeze(2), b3.unsqueeze(2)), dim=2)#[b, 3, 3]
                features = R.bmm(features)
                if(torch.isnan(features).sum()):
                    print(F'R6 = {R6}')
            new_coords = features
            if self.extra_feature_channels == 4:
                normals = inputs[:, 3:6, :]
                centers_coords = coords.mean(dim=2, keepdim=True).expand(-1, -1, n)
                centers_normals = normals.mean(dim=2, keepdim=True).expand(-1, -1, n)
                ppfs = F.ppf(centers_coords, coords, centers_normals, normals)
                # new_nx = batch_base_x.permute(0, 2, 1).bmm(normals) # [b, 1, n]
                # new_ny = batch_base_y.permute(0, 2, 1).bmm(normals) # [b, 1, n]
                # new_nz = batch_base_z.permute(0, 2, 1).bmm(normals)  # [b, 1, n]
                #features = torch.cat((new_x, new_y, new_z, new_nx, new_ny, new_nz), dim=1)
                features = torch.cat((new_x, new_y, new_z, ppfs), dim=1)
            if self.use_new_coords_for_voxel:
                coords = new_coords
        elif self.rot_invariant_preprocess == 'pca':
            # x [b, 3, n]
            # source pca
            s_ba = torch.mean(coords, axis=2, keepdim=True) # [b, 3, 1]
            s = coords - s_ba  # [b, 3, n]
            su, _, _ = torch.svd(s) # [b, 3, 3]
            features = su.permute(0,2,1).bmm(s) # [b, 3, n]
            #print(f'coords={coords}\nfeatures={features}')
            #print(f'features = {features}\ncoords={coords}')
            # FourClassifyscore = self.FourClassify(self.FourFeature(s).max(dim=2).values)
            # #return FourClassifyscore
            # reflect = FourClassifyscore.softmax(dim=1).max(dim=1).indices # [b,]
            # a = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]]).to(s) # [4, 2]
            # a = a.unsqueeze(0).expand(b, -1, -1) # [b, 4, 2]
            # #print(a.shape)
            # b = a.gather(1, reflect.unsqueeze(1).unsqueeze(2).expand(-1, -1, 2)) # [b, 1, 2]
            # sb32 = su[:, :, :2] * b
            # sb31 = sb32[:, :, 0].cross(sb32[:, :, 1]).unsqueeze(2)
            # sb33 = torch.cat((sb32, sb31), dim=2) # [b, 3, 3]
            # after_s = sb33.permute(0,2,1).bmm(s) # [b, 3, n]
            # features = after_s
            # coords = after_s

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
        if self.with_local_feat=='ppf':
           # 已经除去center自己
            assert inputs.shape[1]>=6
            normals = inputs[:, 3:6, :]
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
        elif self.with_local_feat=='fpfh':
            assert inputs.shape[1]>=6
            local_fpfh = []
            for i in range(inputs.shape[0]):
                inputs_numpy = inputs.cpu().numpy()
                points = o3d.utility.Vector3dVector(inputs_numpy[i, :3, :].T)
                normals = o3d.utility.Vector3dVector(inputs_numpy[i, 3:, :].T)
                pcd0 = o3d.geometry.PointCloud()
                pcd0.points = points
                pcd0.normals = normals
                fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd0, o3d.geometry.KDTreeSearchParamRadius(radius=0.3))
                local_fpfh.append(torch.from_numpy(fpfh.data.astype(np.float32)))
            local_features = self.fuser(torch.stack(local_fpfh, 0).cuda())
            features = torch.cat((features, local_features), dim=1)
        elif self.with_local_feat=='change_coords':
            center_coords = coords
            m = center_coords.shape[2]
            neighbor_coords = self.grouper(coords, center_coords) # [b, 3, k, m=n]
            norm_neighbor_coords = neighbor_coords - neighbor_coords.mean(dim=2, keepdim=True) # [b, 3, k, m=n]
            rank = torch.argsort(norm_neighbor_coords.norm(dim=1), dim=1, descending=True) #[b, k, m=n]
            rerank_norm_neighbor_coords = norm_neighbor_coords.gather(2, rank.unsqueeze(1).expand(-1, 3, -1, -1)) # [b, 3, k, m=n]
            mask = rerank_norm_neighbor_coords.norm(dim=1, keepdim=True) < 1e-5 # [b, 1, k, m=n]
            rerank_norm_neighbor_coords /= (rerank_norm_neighbor_coords.norm(dim=1, keepdim=True) + 1e-20)
            #print(rerank_norm_neighbor_coords.norm(dim=1).permute(0, 2, 1))
            base_x = rerank_norm_neighbor_coords[:, :, 0, :] # [b, 3, m=n]
            j = 1
            base_y = rerank_norm_neighbor_coords[:, :, j, :] # [b, 3, m=n]
            #assert (rerank_norm_neighbor_coords.norm(dim=1) < 1e-5).float().sum()<=0
            lamda = (base_x * base_y).sum(dim=1) # [b, m=n]
            prod_mask = ((lamda > 0.9) | (lamda < -0.9)).unsqueeze(1).expand(-1, 3, -1) # [b, 3, m=n]
            norm_mask = (base_y.norm(dim=1, keepdim=True) < 1e-5).expand(-1, 3, -1) # [b, 3, m=n]
            while (j < self.neighbor_num-1) and (prod_mask.float().sum()>0 or norm_mask.float().sum()>0):
                j += 1
                if prod_mask.float().sum()>0:
                    base_y[prod_mask] = rerank_norm_neighbor_coords[:, :, j, :][prod_mask]
                if norm_mask.float().sum()>0:
                    base_y[norm_mask] = rerank_norm_neighbor_coords[:, :, j, :][norm_mask]
                #assert (base_y.norm(dim=1) < 1e-5).float().sum()<=0
                lamda = (base_x * base_y).sum(dim=1) # [b, m=n]
                prod_mask = ((lamda > 0.9) | (lamda < -0.9)).unsqueeze(1).expand(-1, 3, -1) # [b, 3, m=n]
            
            # orthogonality
            try:    
                assert prod_mask.float().sum()<=0
                ort_base_y = base_y - base_x*((base_x*base_y).sum(dim=1, keepdim=True)) # [b, 3, m=n]
            except:
                ort_base_y = base_y
            assert (ort_base_y.norm(dim=1) < 1e-5).float().sum()<=0
            base_y = ort_base_y/(ort_base_y.norm(dim=1, keepdim=True)+1e-10)
            # 别忘了norm
            base_z = base_x.cross(base_y, dim=1) # [b, 3, m=n]
            base_z = base_z/base_z.norm(dim=1, keepdim=True)
            new_x = base_x.unsqueeze(2).mul(norm_neighbor_coords).sum(1, keepdim=True) # [b, 1, k, n]
            new_y = base_y.unsqueeze(2).mul(norm_neighbor_coords).sum(1, keepdim=True) # [b, 1, k, n]
            new_z = base_z.unsqueeze(2).mul(norm_neighbor_coords).sum(1, keepdim=True) # [b, 1, k, n]
            local_new_xyz = torch.cat((new_x, new_y, new_z), dim=1) # [b, 3, k, n]
            local_features = self.fuser(local_new_xyz).max(dim=2).values
            features = torch.cat((features, local_features), dim=1)
        
        #return features
        for i in range(len(self.point_features)):
            features, _ = self.point_features[i]((features, coords))
            #featuresList.append(features)
        if self.is_classify:
            # if self.rot_invariant_preprocess == 'pca':
            #     return (self.classifier(features.max(dim=2).values), FourClassifyscore)
            # else:
            #     #print(torch.cat(featuresList, dim=1).shape)
            #     #return self.classifier(torch.cat(featuresList, dim=1).max(dim=2).values) # [B, K]
            #     return self.classifier(features.max(dim=2).values) # [B, K]
            return self.classifier(features.max(dim=2).values) # [B, K]
        else:
            return features#.max(dim=2).values #[B, C, n]