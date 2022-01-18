from utils.open3d_func import *
import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils.random_choice import randchoice, farthest_point_sample

class _ModelNet40Dataset(Dataset):
    """ [Princeton ModelNet](http://modelnet.cs.princeton.edu/) """
    def __init__(self, datadir, partition, shapenum, num_points, sample_method='random', normalize=True, with_normals=True, random_rot=False):
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
        #print(self.samples)
    def __getitem__(self, index):
        sample, target = self.samples[index]
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
        points = pc_normal[idx, :3].astype(np.float32) # n, 3
        normals = pc_normal[idx, 3:].astype(np.float32) # n, 3
        if self.normalize:
            points -= np.mean(points, axis=0, keepdims=True)
        if self.random_rot:
            if self.with_normals:
                #print(f'before={points}')
                trans, points, normals = random_rotation(points, normals)
                pcd = np.concatenate((points, normals), axis=1)
                #print(f'trans={trans}')
                #print(f'after={points}')
            else:
                _, points = random_rotation(points)
                pcd = points
        else:
            if self.with_normals:
                pcd = np.concatenate((points, normals), axis=1)
            else:
                pcd = points
        
        return pcd.T, target # (3/6, n), int
    
    def __len__(self):
        return len(self.samples)
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
            #print(root, filenametxt)
            for line in f:
                filename = line.strip()
                #print(filename)
                classname = filename[:-5]
                #print(classname)
                target = class_to_idx[classname]
                sample = os.path.join(classname, filename)
                samples.append((sample, target))
        
        return samples

class ModelNet40(dict):
    def __init__(self, root, shapenum:10/40, num_points, split, sample_method='random', normalize=True, with_normals=True, random_rot={'train':False, 'valid':False, 'test':False}):
        for s in split:
            k = s
            if s=='valid':
                s = 'test'
            self[k] = _ModelNet40Dataset(root, s, shapenum, num_points, sample_method=sample_method, normalize=normalize, with_normals=with_normals, random_rot=random_rot[k])

class MeterModelNet40:
    def __init__(self):
        self.pred = 0
        self.num = 0
    def update(self, outputs:torch.Tensor, targets:torch.Tensor):
        with torch.no_grad():
            pred = F.softmax(outputs, dim=1).argmax(1) # (b, )
            self.pred += (pred==targets).sum()
            #print(f'pred = {pred}\ntargets = {targets}')
            
            self.num += outputs.shape[0] 
            #print(self.pred, self.num)
    def compute(self):
        return self.pred.item()/self.num

if __name__ == '__main__':
    pc = torch.rand(1000, 3)
    m=50
    start_idx = torch.randint(pc.shape[0], (1, )).long()
    pts = farthest_point_sample(pc, m, start_idx)
    pts2, idx = farthest_point_sample2(pc, m, start_idx)
    print(f'pts = {pts}\npts2 = {pts2}\nidx = {idx}')