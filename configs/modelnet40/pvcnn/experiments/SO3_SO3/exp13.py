'''
Author: your name
Date: 2022-01-18 10:04:35
LastEditTime: 2022-01-18 17:07:43
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /exp1/configs/modelnet40/pvcnn/experiments/SO3_SO3/exp13.py
'''
from utils.config import configs
import os
configs.exp_name = 'mn40-r-n-sph_dg-cc-ec0-ppf_cos-se-random-1024'
configs.model.point_kernel_formal = 'dgcnn_kernel' # pointnet_kernel, dgcnn_kernel
configs.model.voxel_shape = 'spherical' # cube, spherical

configs.dataset.with_normals = True
configs.model.extra_feature_channels = 0
configs.model.rot_invariant_preprocess = 'change_coords'
configs.model.with_local_feat = 'ppf'
configs.model.with_transform_fine_tune=False
configs.model.use_new_coords_for_voxel=False
configs.model.with_coeff = True
configs.model.with_se = True

configs.train.ckpt_dir = os.path.join('checkpoint', configs.exp_name)
configs.train.common_ckpt_path = os.path.join(configs.train.ckpt_dir, 'common.ckpt.pth')
configs.train.best_ckpt_paths = os.path.join(configs.train.ckpt_dir, '{}.best.ckpt.pth')
configs.train.logfile = os.path.join(configs.train.ckpt_dir, 'train.log')