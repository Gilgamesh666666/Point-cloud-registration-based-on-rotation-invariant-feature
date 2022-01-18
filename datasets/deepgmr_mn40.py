'''
Author: your name
Date: 2020-11-23 22:23:18
LastEditTime: 2022-01-18 11:22:40
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /exp1/datasets/deepgmr_mn40.py
'''
from utils.open3d_func import *
import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils.random_choice import randchoice, farthest_point_sample
from o3d_tools import visualize_tools as vt
from tqdm import tqdm
import time
import h5py
def offical_project(data):
    # pose = np.eye(4)
    # R = random_rotation()
    # pose[:3, :3] = R
    # points = np.dot(points, R.T)
    points = data[...,:3]
    d = np.ones((500, 500))
    ids = -np.ones((500, 500), dtype=np.int)
    #print(np.min(points, axis=0))
    for i, point in enumerate(points):
        x = int((point[0] + 0.5) / 0.02)
        y = int((point[1] + 0.5) / 0.02)
        z = point[2] + 0.5
        if z < d[x, y]:
            d[x, y] = z
            ids[x, y] = i
    ids = np.ravel(ids[ids > 0])
    #points = points[ids]
    #pose[:3, 3] = -points.mean(axis=0)
    #points -= points.mean(axis=0)
    return data[ids]
class TestData(Dataset):
    def __init__(self, path, n_points):
        super(TestData, self).__init__()
        with h5py.File(path, 'r') as f:
            self.source = f['source'][...]
            self.target = f['target'][...]
            self.transform = f['transform'][...]
        self.n_points = n_points
    def __getitem__(self, index):
        index = 190
        pc1 = self.source[index][:self.n_points]
        pc2 = self.target[index][:self.n_points]
        normals1 = get_normals(pc1)
        normals2 = get_normals(pc2)
        transform = self.transform[index]
        #print(normals1.shape)
        pc1, pc2 = pc1.astype('float32'), pc2.astype('float32')
        pcd1 = np.concatenate((pc1, normals1), axis=1)
        pcd2 = np.concatenate((pc2, normals2), axis=1)
        #pcd1 = offical_project(pcd1)
        #pcd2 = offical_project(pcd2)
        return (pcd1.T, pcd2.T), (pcd1[...,:3], pcd2[...,:3],transform.astype('float32'))

    def __len__(self):
        return self.transform.shape[0]
def getdataset(path, n_points):
    return {'test':TestData(path, n_points)}

def test_registration(model, dataloader, meters, config):
    import logging
    logfile = f'{config.model.point_kernel_formal}_{config.cn}_'+config.evaluate.meters['eval-acc_{}'].func+'.log'
    BASIC_FORMAT = '[%(levelname)s] %(asctime)s:%(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(filename=logfile, format=BASIC_FORMAT, datefmt=DATE_FORMAT, level=logging.DEBUG)
    model.eval()
    results = {}
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='test', ncols=0):
        #for inputs, targets in dataloader:
            #print(dataloader.dataset)
            pc1, pc2 = inputs
            #points, trans_points, trans = targets
            feat1 = model(pc1.cuda())
            feat2 = model(pc2.cuda())
            for meter in meters.values():
                meter.update((feat1.cpu().numpy(), feat2.cpu().numpy()), targets)
        
        for k, meter in meters.items():
            results[k] = meter.compute()
            if isinstance(results[k], dict):
                for name, value in results[k].items():
                    logging.info(f'results[{k}][{name}] = {value}')
            else:
                logging.info(f'results[{k}] = {results[k]}')
    return results



class MeterModelNet40_registration:
    def __init__(self, func):
        self.rre = 0
        self.rte = 0
        self.num = 0
        self.succ = 0
        self.rmse = 0
        self.rmse_succ = 0
        self.reg_time = 0
        self.rot_thresh = 1e-05
        self.rmse_thresh = 0.2
        self.translate_thresh = 0.005
        self.func = func
    def update(self, output:torch.Tensor, target:torch.Tensor):
        with torch.no_grad():
            feat1, feat2 = output
            pt1, pt2, gt_trans = target # (b,n,3),(b,n,3),(b,4,4)
            pt1, pt2, gt_trans = pt1.cpu().numpy(), pt2.cpu().numpy(), gt_trans.cpu().numpy()
            
            #print(feat1.shape, feat2.shape, pt1.shape, pt2.shape, gt_trans.shape)
            for i in range(pt1.shape[0]):
                est_trans, reg_time = self.register_one_pair(pt1[i], pt2[i], feat1[i].T, feat2[i].T, func=self.func)
                rotError, translateError = self.RE_TE_one_pair(gt_trans[i], est_trans)
                est_trans_pt = apply_transform_2dim_numpy(pt1[i], est_trans)
                gt_trans_pt = apply_transform_2dim_numpy(pt1[i], gt_trans[i])
                rmse = np.mean(np.linalg.norm((est_trans_pt - gt_trans_pt),axis=1))
                
                # #print("visual before")
                # pcd1 = vt.visualize_pcd(pt1[i], vt.FRAG1_COLOR)
                # pcd2 = vt.visualize_pcd(pt2[i], vt.FRAG2_COLOR)
                # o3d.visualization.draw_geometries(window_name='before', geometry_list=[pcd1, pcd2])
                # #print("visual after")
                # pcd1 = vt.visualize_pcd(pt2[i], vt.FRAG1_COLOR)
                # pcd2 = vt.visualize_pcd(est_trans_pt, vt.FRAG2_COLOR)
                # o3d.visualization.draw_geometries(window_name='Ours', geometry_list=[pcd1, pcd2])
                # #print("visual gt")
                # pcd1 = vt.visualize_pcd(pt2[i], vt.FRAG1_COLOR)
                # pcd2 = vt.visualize_pcd(gt_trans_pt, vt.FRAG2_COLOR)
                # o3d.visualization.draw_geometries(window_name='GT', geometry_list=[pcd1, pcd2])

                if rotError<self.rot_thresh and translateError<self.translate_thresh:
                    self.succ += 1
                if rmse<self.rmse_thresh:
                    self.rmse_succ += 1
                self.rre += rotError
                self.rte += translateError
                self.rmse += rmse
                self.reg_time += reg_time
                self.num += 1
    def compute(self):
        return {'succ':self.succ/self.num, 'rre':self.rre/self.num, 'rte':self.rte/self.num, 'rmse':self.rmse/self.num, 'reg_time':self.reg_time/self.num, 'rmse_succ':self.rmse_succ/self.num}
    def RE_TE_one_pair(self, gt, est):
        import math
        # np [4, 4], [4, 4]
        gt_R = gt[:3, :3] # [3, 3]
        est_R = est[:3, :3] # [3, 3]
        A = (np.trace(np.dot(gt_R.T, est_R)) - 1)/2
        if A > 1:
            A = 1
        elif A < -1:
            A = -1
        rotError = math.degrees(math.fabs(math.acos(A))) # degree
        translateError = np.linalg.norm(gt[:3, 3] - est[:3, 3]) # norm
        return rotError, translateError
    def register_one_pair(self, xyz, xyz_corr, feat, feat_corr, func='teaserpp'):
        if func=='ransac':
            trans, reg_time = register_trad_one_pair(xyz, xyz_corr, feat, feat_corr, func='ransac', max_iter=1000, max_val=500, voxel_size=0.08)
        elif func=='fgr':
            trans, reg_time = register_trad_one_pair(xyz, xyz_corr, feat, feat_corr, func='fgr', voxel_size=0.08)
        elif func=='icp':
            trans, reg_time = register_trad_one_pair(xyz, xyz_corr, feat, feat_corr, func='icp', voxel_size=0.08)
        elif func=='teaserpp':
            NOISE_BOUND = 0.02
            try:
                import teaserpp_python
            except:
                print('please install TEASER++')
                exit(-1)
            def compose_mat4_from_teaserpp_solution(solution):
                """
                Compose a 4-by-4 matrix from teaserpp solution
                """
                s = solution.scale
                rotR = solution.rotation
                t = solution.translation
                T = np.eye(4)
                T[0:3, 3] = t
                R = np.eye(4)
                R[0:3, 0:3] = rotR
                M = T.dot(R)

                if s == 1:
                    M = T.dot(R)
                else:
                    S = np.eye(4)
                    S[0:3, 0:3] = np.diag([s, s, s])
                    M = T.dot(R).dot(S)

                return M
            idx1, idx2 = self.find_correspondence_one_pair(feat, feat_corr)
            source = xyz[idx1].T
            target = xyz_corr[idx2].T
            """
            Use TEASER++ to perform global registration
            """
            # Prepare TEASER++ Solver
            solver_params = teaserpp_python.RobustRegistrationSolver.Params()
            solver_params.cbar2 = 1
            solver_params.noise_bound = NOISE_BOUND
            solver_params.estimate_scaling = False
            solver_params.rotation_estimation_algorithm = (
                teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
            )
            solver_params.rotation_gnc_factor = 1.4
            solver_params.rotation_max_iterations = 100
            solver_params.rotation_cost_threshold = 1e-12
            #print("TEASER++ Parameters are:", solver_params)
            teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

            # Solve with TEASER++
            start = time.time()
            teaserpp_solver.solve(source, target)
            end = time.time()
            est_solution = teaserpp_solver.getSolution()
            est_mat = compose_mat4_from_teaserpp_solution(est_solution)
            max_clique = teaserpp_solver.getTranslationInliersMap()
            print("Max clique size:", len(max_clique))
            #final_inliers = teaserpp_solver.getTranslationInliers()
            trans = est_mat
            reg_time = end - start
        return trans, reg_time
    def find_correspondence_one_pair(self, feat1, feat2):
        # [n1, c], [n2, c]
        # [n1, n2]
        diff = np.power(np.linalg.norm(feat1, axis=1, keepdims=True),2) + np.power(np.linalg.norm(feat2, axis=1, keepdims=True).T,2) - 2 * np.dot(feat1, feat2.T)
        #print(f'diff={diff}')
        corr_idx1 = np.argmin(diff, axis=1) # [n1]
        corr_idx2 = np.argmin(diff, axis=0) # [n2]
        #return np.arange(len(corr_idx1)), corr_idx1
        mask = (corr_idx2[corr_idx1] == np.arange(corr_idx1.shape[0])) # [n1]
        
        idx2 = corr_idx1[mask] # 2
        idx1 = np.arange(corr_idx1.shape[0])[mask] # 1
        return idx1, idx2

if __name__ == '__main__':
    pc = torch.rand(1000, 3)
    m=50
    start_idx = torch.randint(pc.shape[0], (1, )).long()
    pts = farthest_point_sample(pc, m, start_idx)
    pts2, idx = farthest_point_sample2(pc, m, start_idx)
    print(f'pts = {pts}\npts2 = {pts2}\nidx = {idx}')