'''
Author: your name
Date: 2020-11-20 22:28:34
LastEditTime: 2022-01-18 11:15:44
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /exp1/utils/open3d_func.py
'''
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
import torch
import time
def make_O3d_PointCloud(input_nparr:np.array, color:np.array=None):
    # [n, 3]
    pcd = o3d.geometry.PointCloud()
    assert len(input_nparr.shape) == 2
    assert input_nparr.shape[1] == 3
    pcd.points = o3d.utility.Vector3dVector(input_nparr)
    if color is not None:
        assert color.shape == (3, 1)
        pcd.paint_uniform_color(color)
    return pcd

def make_o3d_PointCloud(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd
def make_o3d_Feature(feat):
    # [n, c]
    feature = o3d.pipelines.registration.Feature()
    feature.data = feat.T
    return feature
def register_trad_one_pair(xyz, xyz_corr, feat, feat_corr, func='ransac', voxel_size=0.08, max_iter=100, max_val=20):
    if func=='ransac':
        #print('use ransac')
        assert voxel_size > 0
        source = make_o3d_PointCloud(xyz)
        target = make_o3d_PointCloud(xyz_corr)
        feature_source = make_o3d_Feature(feat)
        feature_target = make_o3d_Feature(feat_corr)
        start = time.time()
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, feature_source, feature_target, voxel_size,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),4,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, max_val)) # max_iter, max_val:只有过了check才会去validation
        end = time.time()
        trans = result.transformation
        reg_time = end - start
    elif func=='fgr':
        #print('use FGR')
        source = make_o3d_PointCloud(xyz)
        target = make_o3d_PointCloud(xyz_corr)
        feature_source = make_o3d_Feature(feat)
        feature_target = make_o3d_Feature(feat_corr)
        start = time.time()
        reg = o3d.pipelines.registration.registration_fast_based_on_feature_matching(source, target, feature_source, feature_target, o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=voxel_size))
        end = time.time()
        trans = reg.transformation
        reg_time = end - start
    elif func=='icp':
        pcd0 = make_o3d_PointCloud(xyz)
        pcd1 = make_o3d_PointCloud(xyz_corr)
        start = time.time()
        reg = o3d.pipelines.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        end = time.time()
        trans = reg.transformation
        reg_time = end - start
    else:
        print('Only Support ransac and fgr')
    return trans, reg_time

def get_normals(pcd:o3d.geometry.PointCloud or np.array):
    if isinstance(pcd, np.ndarray):
        pcd = make_O3d_PointCloud(pcd)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=0.1))
    pcd.orient_normals_towards_camera_location()
    pcd.normalize_normals()
    return np.asarray(pcd.normals,dtype=np.float32)

def random_rotation(points, normals=None, max_degree=360, max_amp=3, seed=0):
    # inputs:[N, 3], None/[N, 3], [N, 3]
    np.random.seed(seed)
    x, degree, amp = np.random.rand(6), np.random.rand(1)*max_degree*np.pi/180, np.random.rand(1)*max_amp
    w, v= x[:3], x[3:]
    w, v = w/np.linalg.norm(w), v/np.linalg.norm(v)
    w *= degree
    v *= amp
    r = Rotation.from_rotvec(w)
    points = r.apply(points) + v[np.newaxis,:]
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = v
    if normals is not None:
        normals = r.apply(normals)
        return T, points.astype(np.float32), normals.astype(np.float32)
    else:
        return T, points.astype(np.float32)

def apply_transform_2dim_numpy(pts:np.array, trans:np.array, with_translate=True):
    # (n, 3) (4, 4)
    R = trans[:3, :3]
    t = trans[:3, 3]
    #r = Rotation.from_matrix(R)
    if with_translate:
        return pts.dot(R.T) + t[np.newaxis,:]
    else:
        return pts.dot(R.T)

def apply_transform_3dim_numpy(pts:np.array, trans:np.array, with_translate=True):
    # (b, n, 3) (b, 4, 4)
    R = trans[:, :3, :3] # [b, 3, 3]
    t = trans[:, :3, 3] # [b, 3]

    if with_translate:
        return np.einsum("ijk, ilk -> ijl", pts, R) + t[:, np.newaxis, :]
    else:
        return np.einsum("ijk, ilk -> ijl", pts, R)

def apply_transform_3dim_torch(pts, trans, with_translate=True):
    # (b, 3, n) (b, 4, 4)
    R = trans[:, :3, :3] # (b, 3, 3)
    t = trans[:, :3, 3].unsqueeze(2) # (b, 3, 1)
    if with_translate:
        return R.bmm(pts) + t # [b, 3, n]
    else:
        return R.bmm(pts) # [b, 3, n]

if __name__ == '__main__':
    pts = np.random.rand(10, 3)
    T = np.eye(4)
    T[:3, 3] = np.random.rand(3)
    print(pts)
    pts_rot = apply_transform_2dim_numpy(pts, T)
    print(pts)
    print(pts_rot)