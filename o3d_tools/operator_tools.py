import os
import open3d
import numpy as np

def read_pcd_ply(file):
    # [n, 3] or str
    if os.path.exist(file):
        pcd = open3d.io.read_pcd_ply(file)
    else:
        raise ValueError
    return np.asarray(pcd.points)