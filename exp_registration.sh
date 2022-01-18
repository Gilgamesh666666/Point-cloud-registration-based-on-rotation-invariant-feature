###
 # @Author: your name
 # @Date: 2020-11-26 21:47:38
 # @LastEditTime: 2022-01-18 15:32:06
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /exp1/run.sh
### 
# python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_dg/clean_fgr.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_dg-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
# python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_dg/clean_ransac.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_dg-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
# python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_dg/clean_teaserpp.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_dg-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
#python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_dg/noise_fgr.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_dg-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
#python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_dg/noise_ransac.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_dg-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
#python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_dg/noise_teaserpp.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_dg-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
# python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_dg/icl_nuim_fgr.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_dg-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
# python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_dg/icl_nuim_ransac.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_dg-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
# python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_dg/icl_nuim_teaserpp.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_dg-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth


# python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_pt/clean_fgr.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_pt-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
# python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_pt/clean_ransac.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_pt-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
# python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_pt/clean_teaserpp.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_pt-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
#python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_pt/noise_fgr.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_pt-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
#python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_pt/noise_ransac.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_pt-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
#python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_pt/noise_teaserpp.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_pt-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
# python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_pt/icl_nuim_fgr.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_pt-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
# python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_pt/icl_nuim_ransac.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_pt-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth
python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/deepgmr_mn40_cu_pt/icl_nuim_teaserpp.py --device 0 --evaluate --eval_ckpt_pth checkpoint/mn40-n-r-cu_pt-cc-withlocalppf-with_se-random-1024/train-acc_valid.best.ckpt.pth


# python train.py --config configs/modelnet40/pvcnn/experiments/SO3_SO3/exp14.py --device 0