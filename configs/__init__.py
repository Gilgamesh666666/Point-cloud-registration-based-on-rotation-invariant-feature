from utils.config import Config, configs
import torch.optim as optim
import os

configs.seed = 0
configs.deterministic = False
configs.parallel = True

configs.dataloader = Config()
configs.dataloader.pin_memory = False
