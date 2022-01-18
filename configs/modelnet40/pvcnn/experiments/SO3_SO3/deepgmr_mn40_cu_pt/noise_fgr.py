'''
Author: your name
Date: 2020-11-26 21:42:52
LastEditTime: 2020-11-26 21:46:33
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /exp1/configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_dg/clean_teaserpp.py
'''
from utils.config import configs
configs.evaluate.meters['eval-acc_{}'].func = 'fgr'
configs.evaluate.other_dataset.path='/home/zebai/exp1/data/test/modelnet_noisy.h5'
configs.cn = 'noisy'
