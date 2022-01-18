
from utils.config import configs

configs.dataset.num_points = 1024
configs.dataloader.batch_size = {'train':16, 'valid':32, 'test':32}
configs.dataloader.num_workers = 8

configs.dataset.sample_method = 'random'
configs.dataset.random_rot = {'train':True, 'valid':True, 'test':True}
configs.train.optimizer.weight_decay = 1e-6
