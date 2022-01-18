
from utils.config import configs
import os
configs.exp_name = 'sn-sph-dg-change_coords-withlocal-1024'
configs.model.point_kernel_formal = 'dgcnn_kernel' # pointnet_kernel, dgcnn_kernel
configs.model.voxel_shape = 'spherical' # cube, spherical

configs.model.width_multiplier = 1
configs.model.voxel_resolution_multiplier = 1

configs.dataloader.batch_size = {'train':16, 'valid':32, 'test':32}
configs.dataloader.num_workers = 16

# with voxel-PPF
configs.dataset.with_normal = True
configs.model.extra_feature_channels = 4

configs.model.rot_invariant_preprocess = 'change_coords'
configs.model.with_local_feat = True
configs.dataset.with_random_rot = True

configs.train.ckpt_dir = os.path.join('checkpoint', configs.exp_name)
configs.train.common_ckpt_path = os.path.join(configs.train.ckpt_dir, 'common.ckpt.pth')
configs.train.best_ckpt_paths = os.path.join(configs.train.ckpt_dir, '{}.best.ckpt.pth')
configs.train.logfile = os.path.join(configs.train.ckpt_dir, 'train.log')

configs.train.optimizer.weight_decay = 1e-6