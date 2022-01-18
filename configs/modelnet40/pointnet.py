
from utils.config import Config, configs
from PVCNN.models.pointnet_classify import pointnet
import os
import torch.optim as optim
configs.dataset.num_points = 1024

configs.dataloader.batch_size = {'train':32, 'valid':32, 'test':32}
configs.dataloader.num_workers = 32

configs.model = Config(pointnet)
configs.model.inchannel = 6
configs.model.outchannels = [64, 64, 128]
configs.model.num_class = 40

configs.exp_name = 'mn40-pt-1024'

#configs.dataloader.batch_size = {'train':16, 'valid':32, 'test':32}
configs.dataloader.num_workers = 16

# with voxel-PPF
configs.dataset.with_normals = True
configs.dataset.random_rot = {'train':True, 'valid':True, 'test':True}

configs.train.ckpt_dir = os.path.join('checkpoint', configs.exp_name)
configs.train.common_ckpt_path = os.path.join(configs.train.ckpt_dir, 'common.ckpt.pth')
configs.train.best_ckpt_paths = os.path.join(configs.train.ckpt_dir, '{}.best.ckpt.pth')
configs.train.logfile = os.path.join(configs.train.ckpt_dir, 'train.log')

configs.train.optimizer.weight_decay = 1e-6
configs.train.valid_interval = 1
configs.train.num_epochs = 250

configs.train.scheduler = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs