'''
Author: your name
Date: 2020-11-26 21:42:52
LastEditTime: 2022-01-18 10:57:10
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /exp1/configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_dg/clean_teaserpp.py
'''
from utils.config import configs
configs.evaluate.meters['eval-acc_{}'].func = 'teaserpp'
configs.evaluate.other_dataset.path='/home/zebai/exp1/data/test/modelnet_clean.h5'
configs.cn = 'clean'
