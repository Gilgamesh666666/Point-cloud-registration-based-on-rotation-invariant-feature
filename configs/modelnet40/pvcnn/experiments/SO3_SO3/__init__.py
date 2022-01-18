'''
Author: your name
Date: 2022-01-18 10:04:35
LastEditTime: 2022-01-18 19:35:12
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /exp1/configs/modelnet40/pvcnn/experiments/SO3_SO3/__init__.py
'''

from utils.config import configs

configs.dataset.num_points = 1024
configs.dataloader.batch_size = {'train':16, 'valid':32, 'test':32}
configs.dataloader.num_workers = 8
configs.dataloader.drop_last = True

configs.dataset.sample_method = 'random'
configs.dataset.random_rot = {'train':True, 'valid':True, 'test':True}
configs.train.optimizer.weight_decay = 1e-6
