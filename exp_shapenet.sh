###
 # @Author: your name
 # @Date: 2020-11-21 11:36:09
 # @LastEditTime: 2022-01-18 16:24:27
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /exp1/exp_shapenet.sh
### 
#! /usr/bash/bin
python train.py --config configs/shapenet/pvcnn/rot_with_ppf.py --device 0 --evaluate