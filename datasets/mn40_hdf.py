"""Data loader
"""
import argparse
import logging
import os
from typing import List

import h5py
import numpy as np
import open3d as o3d
from utils.open3d_func import *
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision

import datasets.transforms as Transforms
from datasets.math_torch import se3
from datasets.so3 import dcm2euler
_logger = logging.getLogger()


def get_train_datasets(args: argparse.Namespace):
    train_categories, val_categories = None, None
    if args.train_categoryfile:
        train_categories = [line.rstrip('\n') for line in open(args.train_categoryfile)]
        train_categories.sort()
    if args.val_categoryfile:
        val_categories = [line.rstrip('\n') for line in open(args.val_categoryfile)]
        val_categories.sort()

    train_transforms, val_transforms = get_transforms(args.noise_type, args.rot_mag, args.trans_mag,
                                                      args.num_points, args.partial)
    _logger.info('Train transforms: {}'.format(', '.join([type(t).__name__ for t in train_transforms])))
    _logger.info('Val transforms: {}'.format(', '.join([type(t).__name__ for t in val_transforms])))
    train_transforms = torchvision.transforms.Compose(train_transforms)
    val_transforms = torchvision.transforms.Compose(val_transforms)

    if args.dataset_type == 'modelnet_hdf':
        train_data = ModelNetHdf(args.dataset_path, subset='train', categories=train_categories,
                                 transform=train_transforms)
        val_data = ModelNetHdf(args.dataset_path, subset='test', categories=val_categories,
                               transform=val_transforms)
    else:
        raise NotImplementedError

    return train_data, val_data


def get_test_datasets(args: argparse.Namespace):
    test_categories = None
    if args.test_category_file:
        test_categories = [line.rstrip('\n') for line in open(args.test_category_file)]
        test_categories.sort()

    _, test_transforms = get_transforms(args.noise_type, args.rot_mag, args.trans_mag, args.num_points, args.partial)
    _logger.info('Test transforms: {}'.format(', '.join([type(t).__name__ for t in test_transforms])))
    test_transforms = torchvision.transforms.Compose(test_transforms)

    if args.dataset_type == 'modelnet_hdf':
        test_data = ModelNetHdf(args.dataset_path, subset='test', categories=test_categories, transform=test_transforms)
    else:
        raise NotImplementedError

    return {'test':test_data}


def get_transforms(noise_type: str,
                   rot_mag: float = 45.0, trans_mag: float = 0.5,
                   num_points: int = 1024, partial_p_keep: List = None):
    """Get the list of transformation to be used for training or evaluating RegNet

    Args:
        noise_type: Either 'clean', 'jitter', 'crop'.
          Depending on the option, some of the subsequent arguments may be ignored.
        rot_mag: Magnitude of rotation perturbation to apply to source, in degrees.
          Default: 45.0 (same as Deep Closest Point)
        trans_mag: Magnitude of translation perturbation to apply to source.
          Default: 0.5 (same as Deep Closest Point)
        num_points: Number of points to uniformly resample to.
          Note that this is with respect to the full point cloud. The number of
          points will be proportionally less if cropped
        partial_p_keep: Proportion to keep during cropping, [src_p, ref_p]
          Default: [0.7, 0.7], i.e. Crop both source and reference to ~70%

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """

    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]

    if noise_type == "clean":
        # 1-1 correspondence for each point (resample first before splitting), no noise
        train_transforms = [Transforms.Resampler(num_points),
                            Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.FixedResampler(num_points),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag)]#,
                           #Transforms.ShufflePoints()]

    elif noise_type == "jitter":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]

    elif noise_type == "crop":
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomCrop(partial_p_keep),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomCrop(partial_p_keep),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]
    else:
        raise NotImplementedError

    return train_transforms, test_transforms

def test_registration(model, dataloader, meters):
    model.eval()
    results = {}
    with torch.no_grad():
        for data in tqdm(dataloader, desc='test', ncols=0):
        #for inputs, targets in dataloader:
            #print(dataloader.dataset)
            dict_all_to_device(data, 'cuda')
            pcd1, pcd2 = data['points_ref'].permute(0, 2, 1), data['points_src'].permute(0, 2, 1)
            #points, trans_points, trans = targets
            feat1 = model(pcd1.cuda())
            feat2 = model(pcd2.cuda())
            for meter in meters.values():
                meter.update((feat1.detach().cpu().numpy(), feat2.detach().cpu().numpy()), data)
        
        for k, meter in meters.items():
            results[k] = meter.compute()
            if isinstance(results[k], dict):
                for name, value in results[k].items():
                    print(f'results[{k}][{name}] = {value}')
            else:
                print(f'results[{k}] = {results[k]}')
    return results
class ModelNetHdf(Dataset):
    def __init__(self, dataset_path: str, subset: str = 'train', categories: List = None, transform=None):
        """ModelNet40 dataset from PointNet.
        Automatically downloads the dataset if not available

        Args:
            dataset_path (str): Folder containing processed dataset
            subset (str): Dataset subset, either 'train' or 'test'
            categories (list): Categories to use
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path

        metadata_fpath = os.path.join(self._root, '{}_files.txt'.format(subset))
        self._logger.info('Loading data from {} for {}'.format(metadata_fpath, subset))

        if not os.path.exists(os.path.join(dataset_path)):
            self._download_dataset(dataset_path)

        with open(os.path.join(dataset_path, 'shape_names.txt')) as fid:
            self._classes = [l.strip() for l in fid]
            self._category2idx = {e[1]: e[0] for e in enumerate(self._classes)}
            self._idx2category = self._classes

        with open(os.path.join(dataset_path, '{}_files.txt'.format(subset))) as fid:
            h5_filelist = [line.strip() for line in fid]
            h5_filelist = [x.replace('data/modelnet40_ply_hdf5_2048/', '') for x in h5_filelist]
            h5_filelist = [os.path.join(self._root, f) for f in h5_filelist]

        if categories is not None:
            categories_idx = [self._category2idx[c] for c in categories]
            self._logger.info('Categories used: {}.'.format(categories_idx))
            self._classes = categories
        else:
            categories_idx = None
            self._logger.info('Using all categories.')

        self._data, self._labels = self._read_h5_files(h5_filelist, categories_idx)
        # self._data, self._labels = self._data[:32], self._labels[:32, ...]
        self._transform = transform
        self._logger.info('Loaded {} {} instances.'.format(self._data.shape[0], subset))

    def __getitem__(self, item):
        sample = {'points': self._data[item, :, :], 'label': self._labels[item], 'idx': np.array(item, dtype=np.int32)}

        if self._transform:
            sample = self._transform(sample)

        return sample

    def __len__(self):
        return self._data.shape[0]

    @property
    def classes(self):
        return self._classes

    @staticmethod
    def _read_h5_files(fnames, categories):

        all_data = []
        all_labels = []

        for fname in fnames:
            f = h5py.File(fname, mode='r')
            data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1)
            labels = f['label'][:].flatten().astype(np.int64)

            if categories is not None:  # Filter out unwanted categories
                mask = np.isin(labels, categories).flatten()
                data = data[mask, ...]
                labels = labels[mask, ...]

            all_data.append(data)
            all_labels.append(labels)

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels

    @staticmethod
    def _download_dataset(dataset_path: str):
        os.makedirs(dataset_path, exist_ok=True)

        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget {}'.format(www))
        os.system('unzip {} -d .'.format(zipfile))
        os.system('mv {} {}'.format(zipfile[:-4], os.path.dirname(dataset_path)))
        os.system('rm {}'.format(zipfile))

    def to_category(self, i):
        return self._idx2category[i]
def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise NotImplementedError
def dict_all_to_device(tensor_dict, device):
    """Sends everything into a certain device """
    for k in tensor_dict:
        if isinstance(tensor_dict[k], torch.Tensor):
            tensor_dict[k] = tensor_dict[k].to(device)

from o3d_tools import visualize_tools as vt
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
        self.metrics_for_iter = {'r_mse': [],
                'r_mae': [],
                't_mse': [],
                't_mae': [],
                'err_r_deg': [],
                'err_t': [],
                'chamfer_dist': []}
    @staticmethod
    def compute_metrics(data, pred_transforms):
        def square_distance(src, dst):
            return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

        with torch.no_grad():
            pred_transforms = pred_transforms
            
            gt_transforms = data['transform_gt']
            points_src = data['points_src'][..., :3]
            points_ref = data['points_ref'][..., :3]
            points_raw = data['points_raw'][..., :3]
            #print(se3.inverse(gt_transforms))
            # for i in range(min(points_src.shape[0], 5)):
            #     pcd1 = vt.visualize_pcd(points_src[i].cpu().numpy(), vt.FRAG1_COLOR)
            #     pcd2 = vt.visualize_pcd(points_ref[i].cpu().numpy(), vt.FRAG2_COLOR)
            #     open3d.visualization.draw_geometries([pcd1,pcd2])
            # Euler angles, Individual translation errors (Deep Closest Point convention)
            # TODO Change rotation to torch operations
            r_gt_euler_deg = dcm2euler(gt_transforms[:, :3, :3].detach().cpu().numpy(), seq='xyz')
            r_pred_euler_deg = dcm2euler(pred_transforms[:, :3, :3].detach().cpu().numpy(), seq='xyz')
            t_gt = gt_transforms[:, :3, 3]
            t_pred = pred_transforms[:, :3, 3]
            r_mse = np.mean((r_gt_euler_deg - r_pred_euler_deg) ** 2, axis=1)
            r_mae = np.mean(np.abs(r_gt_euler_deg - r_pred_euler_deg), axis=1)
            t_mse = torch.mean((t_gt - t_pred) ** 2, dim=1)
            t_mae = torch.mean(torch.abs(t_gt - t_pred), dim=1)

            # Rotation, translation errors (isotropic, i.e. doesn't depend on error
            # direction, which is more representative of the actual error)
            concatenated = se3.concatenate(se3.inverse(gt_transforms), pred_transforms)
            rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
            residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
            residual_transmag = concatenated[:, :, 3].norm(dim=-1)
            # Modified Chamfer distance
            src_transformed = se3.transform(pred_transforms, points_src)
            ref_clean = points_raw
            src_clean = se3.transform(se3.concatenate(pred_transforms, se3.inverse(gt_transforms)), points_raw)
            dist_src = torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0]
            dist_ref = torch.min(square_distance(points_ref, src_clean), dim=-1)[0]
            chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

            metrics = {
                'r_mse': r_mse,
                'r_mae': r_mae,
                't_mse': to_numpy(t_mse),
                't_mae': to_numpy(t_mae),
                'err_r_deg': to_numpy(residual_rotdeg),
                'err_t': to_numpy(residual_transmag),
                'chamfer_dist': to_numpy(chamfer_dist)
            }

        return metrics


    @staticmethod
    def summarize_metrics(metrics):
        """Summaries computed metrices by taking mean over all data instances"""
        summarized = {}
        for k in metrics:
            if k.endswith('mse'):
                summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
            elif k.startswith('err'):
                summarized[k + '_mean'] = np.mean(metrics[k])
                summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k]**2))
            else:
                summarized[k] = np.mean(metrics[k])

        return summarized
    def update(self, output:torch.Tensor, target:torch.Tensor):
        with torch.no_grad():
            feat1, feat2 = output
            #pt1, pt2, gt_trans = target # (b,n,3),(b,n,3),(b,4,4)
            #pt1, pt2, gt_trans = pt1.cpu().numpy(), pt2.cpu().numpy(), gt_trans.cpu().numpy()
            
            #print(feat1.shape, feat2.shape, pt1.shape, pt2.shape, gt_trans.shape)
            pt1, pt2 = target['points_ref'][:, :, :3], target['points_src'][:, :, :3]
            est_trans_list = []
            for i in range(pt1.shape[0]):
                est_trans, reg_time = self.register_one_pair(pt1[i].detach().cpu().numpy(), pt2[i].detach().cpu().numpy(), feat1[i].T, feat2[i].T, func=self.func)
                est_trans_list.append(est_trans)
                # print("visual")
                # pcd1 = vt.visualize_pcd(pt1[i].detach().cpu().numpy(), vt.FRAG1_COLOR)
                # pcd2 = vt.visualize_pcd(pt2[i].detach().cpu().numpy(), vt.FRAG2_COLOR)
                # o3d.visualization.draw_geometries([pcd1, pcd2])
                #o3d.visualization.draw_geometries([pcd1])
            pred_trans = torch.from_numpy(np.stack(est_trans_list, axis=0).astype(np.float32))
            metrics = self.compute_metrics(target, pred_trans.to('cuda'))
            for k in metrics:
                self.metrics_for_iter[k].append(metrics[k])
    def compute(self):
        return self.summarize_metrics({k: np.concatenate(self.metrics_for_iter[k], axis=0) for k in self.metrics_for_iter})
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