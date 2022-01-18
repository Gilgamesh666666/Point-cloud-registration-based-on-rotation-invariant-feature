
from utils.config import Config, configs
import torch.nn as nn
import torch.optim as optim
from datasets.modelnet40 import ModelNet40, MeterModelNet40

configs.dataset = Config(ModelNet40, split=['train', 'valid', 'test'])
configs.dataset.root = "/media/zebai/T7/Datasets/modelnet40_normal_resampled"
configs.dataset.shapenum = 40

configs.train = Config()

configs.train.optimizer = Config(optim.Adam)
configs.train.optimizer.lr = 1e-3


configs.train.criterion = Config(nn.CrossEntropyLoss)
configs.train.meters = Config()
configs.train.meters['train-acc_{}'] = Config(MeterModelNet40)

configs.evaluate = Config()
configs.evaluate.meters = Config()
configs.evaluate.meters['eval-acc_{}'] = Config(MeterModelNet40)
configs.evaluate.fn = None