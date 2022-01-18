
from utils.config import configs, Config
import os
configs.exp_name = 'mn40-r-n-cu_dg-cc-ec0-withlocalppf-with_se-random-1024'
# configs.model.point_kernel_formal = 'dgcnn_kernel' # pointnet_kernel, dgcnn_kernel
# configs.model.voxel_shape = 'cube' # cube, spherical
# configs.model.extra_feature_channels = 0
# configs.model.rot_invariant_preprocess = 'change_coords'
# configs.model.with_local_feat = 'ppf'
# configs.model.with_transform_fine_tune=False
# configs.model.use_new_coords_for_voxel=False
# configs.model.is_classify = False
configs.model.point_kernel_formal = 'pointnet_kernel' # pointnet_kernel, dgcnn_kernel
configs.model.voxel_shape = 'cube' # cube, spherical

configs.dataset.with_normals = True
configs.model.extra_feature_channels = 4
configs.model.rot_invariant_preprocess = 'change_coords'
configs.model.with_local_feat = 'ppf'
configs.model.with_transform_fine_tune=False
configs.model.use_new_coords_for_voxel=False
configs.model.is_classify = False
# from models.shapenet.pvcnn import PVCNN
# num_classes = 50
# num_shapes = 16
# # model
# configs.model = Config(PVCNN)
# configs.model.num_classes = num_classes
# configs.model.num_shapes = num_shapes
# configs.model.extra_feature_channels = 3

configs.train.ckpt_dir = os.path.join('checkpoint', configs.exp_name)
configs.train.common_ckpt_path = os.path.join(configs.train.ckpt_dir, 'common.ckpt.pth')
configs.train.best_ckpt_paths = os.path.join(configs.train.ckpt_dir, '{}.best.ckpt.pth')
configs.train.logfile = os.path.join(configs.train.ckpt_dir, 'train.log')

configs.dataset.with_normals = True
from datasets.deepgmr_mn40 import getdataset, MeterModelNet40_registration, test_registration
configs.evaluate.meters['eval-acc_{}'] = Config(MeterModelNet40_registration)
configs.evaluate.fn = test_registration
configs.evaluate.other_dataset = Config(getdataset, n_points=1024)
configs.dataloader.batch_size = {'train':32, 'valid':32, 'test':1}
