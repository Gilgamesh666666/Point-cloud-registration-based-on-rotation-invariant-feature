from utils.config import Config, configs
import torch.nn as nn
import torch.optim as optim
from PVCNN.models.shapenet_pvcnn import shapenet_PVCNN
from datasets.shapenet import MeterShapeNet

configs.model = Config(shapenet_PVCNN)
configs.model.blocks = ((64, 1, 32), (128, 2, 16), (512, 1, None), (2048, 1, None))
configs.model.num_classes = 50
configs.model.num_shapes = 16

configs.train = Config()
configs.train.num_epochs = 400

configs.train.optimizer = Config(optim.Adam)
configs.train.optimizer.lr = 1e-3
configs.train.valid_interval = 1
configs.train.scheduler = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs

configs.train.criterion = Config(nn.CrossEntropyLoss)
configs.train.meters = Config()
configs.train.meters['train-iou_{}'] = Config(MeterShapeNet)