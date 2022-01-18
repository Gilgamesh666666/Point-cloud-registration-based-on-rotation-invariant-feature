from PVCNN.models.pvcnn_classify import PVCNN_classifier
from utils.config import Config, configs
import torch.optim as optim

configs.model = Config(PVCNN_classifier)
configs.model.dim_k = 512
configs.model.blocks = ((64, 1, 32), (128, 1, 32), (256, 1, None), (configs.model.dim_k, 1, None))
configs.model.num_classes = 40

configs.model.width_multiplier = 1
configs.model.voxel_resolution_multiplier = 1
configs.model.is_classify = True
configs.train.valid_interval = 1
configs.train.num_epochs = 250

configs.train.scheduler = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs