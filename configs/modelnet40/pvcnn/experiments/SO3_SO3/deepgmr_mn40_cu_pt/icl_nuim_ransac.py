'''
Author: your name
Date: 2020-11-26 22:17:10
LastEditTime: 2022-01-18 11:16:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /exp1/configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_dg/icl_nuim_fgr.py
'''
from utils.config import configs
configs.evaluate.meters['eval-acc_{}'].func = 'ransac'
configs.evaluate.other_dataset.path='/home/zebai/exp1/data/test/icl_nuim_test.h5'
configs.cn = 'icl_nuim'