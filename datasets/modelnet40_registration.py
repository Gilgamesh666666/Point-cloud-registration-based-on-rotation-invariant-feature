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
class _ModelNet40Dataset_registration(Dataset):
    """ [Princeton ModelNet](http://modelnet.cs.princeton.edu/) """
    def __init__(self, datadir, partition, shapenum, num_points, sample_method='random', normalize=True, with_normals=True, random_rot=True, max_degree=360, max_amp=3):
        self.rootdir = datadir
        ptn = partition
        shapenum = shapenum
        self.num_points = num_points
        self.normalize = normalize
        shapename =  f"modelnet{shapenum}_shape_names.txt"
        filenametxt = f"modelnet{shapenum}_{ptn}.txt"
        _, class_to_idx = self.find_classes(self.rootdir, shapename)
        self.samples = self.glob_dataset(self.rootdir, filenametxt, class_to_idx)
        self.with_normals = with_normals
        self.random_rot = random_rot
        self.sample_method = sample_method
        self.max_degree = max_degree
        self.max_amp = max_amp
        self.add_noise = True
        self.diff_sample = True
        self.clip = 0.05
        self.p_keep = 0
        #print(self.samples)
    def __getitem__(self, index):
        sample, target = self.samples[20] #0, 100, 2000
        #print(os.path.join(self.rootdir, sample))
        pc_normal = np.loadtxt(os.path.join(self.rootdir, sample + '.txt'), delimiter=',')
        if self.sample_method == 'random':
            idx = randchoice(pc_normal.shape[0], self.num_points)
        elif self.sample_method == 'fps':
            _, idx = farthest_point_sample(pc_normal.shape[0], self.num_points)
            savePath = os.path.join(self.rootdir, sample + f'_fps_{self.num_points}.npy')
            if os.path.exists(savePath):
                idx = np.load(savePath)
            else:
                np.save(savePath, idx)
        #---------- debug -----------------
        #idx = np.arange(self.num_points)
        #----------------------------------
        # points = pc_normal[idx, :3].astype(np.float32) # n, 3
        # normals = pc_normal[idx, 3:].astype(np.float32) # n, 3
        # For reduce point as RPM-Net and
        if self.diff_sample:
            idx = randchoice(pc_normal.shape[0], 2048)
            points = pc_normal[idx, :3].astype(np.float32) # n, 3
            normals = pc_normal[idx, 3:].astype(np.float32) # n, 3
        else:
            points = pc_normal[idx, :3].astype(np.float32) # n, 3
            normals = pc_normal[idx, 3:].astype(np.float32) # n, 3
        #print(points.shape)
        if self.normalize:
            points -= np.mean(points, axis=0, keepdims=True)
        
        #print(self.max_degree, self.max_amp)
        # Add noise
        if self.clip:
            noise1 = np.clip(0.01 * np.random.randn(points.shape[0], 3), -1*self.clip, self.clip)
            noise2 = np.clip(0.01 * np.random.randn(points.shape[0], 3), -1*self.clip, self.clip)
        else:
            noise1 = 0.01 * np.random.randn(points.shape[0], 3)
            noise2 = 0.01 * np.random.randn(points.shape[0], 3)

        if self.with_normals:
            trans, trans_points, trans_normals = random_rotation(points, normals, max_degree=self.max_degree, max_amp=self.max_amp, seed=0)
            if self.add_noise:
                points += noise1
                trans_points += noise2
            pcd = np.concatenate((points, normals), axis=1)
            trans_pcd = np.concatenate((trans_points, trans_normals), axis=1)
            #print(f'trans={trans}')
        else:
            trans, trans_points = random_rotation(points, max_degree=self.max_degree, max_amp=self.max_amp, seed=0)
            if self.add_noise:
                points += noise1
                trans_points += noise2
            pcd = points
            trans_pcd = trans_points
        if self.p_keep:
            pcd = self.crop(pcd, self.p_keep)
            trans_pcd = self.crop(trans_pcd, self.p_keep)
        # Different sample for source and target:
        if self.diff_sample:
            idx1 = randchoice(pcd.shape[0], self.num_points)
            idx2 = randchoice(trans_pcd.shape[0], self.num_points)
            trans_pcd = trans_pcd[idx2]
            pcd = pcd[idx1]
            points, trans_points = points[idx1], trans_points[idx2]
        
        #print('sample')
            
        return (pcd.T, trans_pcd.T), (pcd[:, :3], trans_pcd[:, :3], trans) # (3/6, n), ((n,3),(n,3),(4,4))
    
    def __len__(self):
        return 1#len(self.samples)
        #return 2
    @staticmethod
    def find_classes(root, class_file_name):
        ''' find ${root}/${class}/* '''
    
        class_file = os.path.join(root, class_file_name)
        with open(class_file, 'r') as f:
            classes =[line.strip() for line in f]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    @staticmethod
    def glob_dataset(root, filenametxt, class_to_idx):
        """ glob ${root}/${class}/${ptns[i]} """
        samples = []
        #class_size = [0 for i in range(len(class_to_idx))]
        with open(os.path.join(root, filenametxt)) as f:
            for line in f:
                filename = line.strip()
                classname = filename[:-5]
                target = class_to_idx[classname]
                sample = os.path.join(classname, filename)
                samples.append((sample, target))
        
        return samples
    def crop(self, points, p_keep):
        def uniform_2_sphere(num: int = None):
            """Uniform sampling on a 2-sphere

            Source: https://gist.github.com/andrewbolster/10274979

            Args:
                num: Number of vectors to sample (or None if single)

            Returns:
                Random Vector (np.ndarray) of size (num, 3) with norm 1.
                If num is None returned value will have size (3,)

            """
            if num is not None:
                phi = np.random.uniform(0.0, 2 * np.pi, num)
                cos_theta = np.random.uniform(-1.0, 1.0, num)
            else:
                phi = np.random.uniform(0.0, 2 * np.pi)
                cos_theta = np.random.uniform(-1.0, 1.0)

            theta = np.arccos(cos_theta)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)

            return np.stack((x, y, z), axis=-1)
        rand_xyz = uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :]

class ModelNet40_registration(dict):
    def __init__(self, root, shapenum:10/40, num_points, split, sample_method='random', normalize=True, with_normals=True, random_rot={'train':False, 'valid':False, 'test':False}, max_degree=360, max_amp=3):
        for s in split:
            k = s
            if s=='valid':
                s = 'test'
            self[k] = _ModelNet40Dataset_registration(root, s, shapenum, num_points, sample_method=sample_method, normalize=normalize, with_normals=with_normals, random_rot=random_rot[k], max_degree=max_degree, max_amp=max_amp)
#features [B, C]
from tqdm import tqdm
import time
def test_registration(model, dataloader, meters):
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
                    print(f'results[{k}][{name}] = {value}')
            else:
                print(f'results[{k}] = {results[k]}')
    return results



class MeterModelNet40_registration:
    def __init__(self):
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
        self.func = 'teaserpp'
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
                
                print("visual before")
                pcd1 = vt.visualize_pcd(pt1[i], vt.FRAG1_COLOR)
                pcd2 = vt.visualize_pcd(pt2[i], vt.FRAG2_COLOR)
                o3d.visualization.draw_geometries([pcd1, pcd2])
                print("visual after")
                pcd1 = vt.visualize_pcd(pt2[i], vt.FRAG1_COLOR)
                pcd2 = vt.visualize_pcd(est_trans_pt, vt.FRAG2_COLOR)
                o3d.visualization.draw_geometries([pcd1, pcd2])
                print("visual gt")
                pcd1 = vt.visualize_pcd(pt2[i], vt.FRAG1_COLOR)
                pcd2 = vt.visualize_pcd(gt_trans_pt, vt.FRAG2_COLOR)
                o3d.visualization.draw_geometries([pcd1, pcd2])
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
    def register_one_pair(self, xyz, xyz_corr, feat, feat_corr, func='ransac'):
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