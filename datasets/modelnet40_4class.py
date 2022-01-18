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
        sample, label = self.samples[index]
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
        points = pc_normal[idx, :3].astype(np.float32) # n, 3
        normals = pc_normal[idx, 3:].astype(np.float32) # n, 3
        if self.normalize:
            points -= np.mean(points, axis=0, keepdims=True)
        if self.random_rot:
            if self.with_normals:
                trans, target, target_normals = random_rotation(points, normals)
                
                pcd = np.concatenate((target, target_normals), axis=1)
                #print(t)
            else:
                trans, target = random_rotation(points)
                pcd = target
            FourClasslabel = self.get4label(points, target, trans)
            labels = (label, FourClasslabel)
        else:
            if self.with_normals:
                pcd = np.concatenate((points, normals), axis=1)
            else:
                pcd = points
            labels = label
        # if self.random_rot:
        #     _, points = random_rotation(points)
        # if self.normalize:
        #     points = points - np.mean(points, axis=0, keepdims=True)
        #     #print(f'points = {points}')
        # if self.with_normals:
        #     normals = get_normals(points)
        #     pcd = np.concatenate((points, normals), axis=1)
        # else:
        #     pcd = points
        #print(self.with_normals)
        return pcd.T, labels # (3/6, n), int
    
    def __len__(self):
        return len(self.samples)
        #return 1024
    def get4label(self, source, target, gTtrans):
        # source pca
        trans = gTtrans[:3, :3]
        n, _ = source.shape
        s_ba = np.mean(source, axis=0, keepdims=True) # [3, 1]
        s = source - s_ba  # [n, 3]
        su, _, _ = np.linalg.svd(s.T) # [3, 3]
        # target pca
        t_ba = np.mean(target, axis=0, keepdims=True) # [3, 1]
        t = target - t_ba # [n, 3]
        tu, _, _ = np.linalg.svd(t.T) # [3, 3]
        #print([np.sign(trans.T.dot(tu)/su)[0, 0], np.sign(trans.T.dot(tu)/su)[0, 1]])
        temp = (1 - np.sign(trans.T.dot(tu)/su))/2
        sign1, sign2 = temp[0, 0], temp[0, 1]
        #print([sign1, sign2])
        label = int(sign1*2 + sign2)
        return label

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
                label = class_to_idx[classname]
                sample = os.path.join(classname, filename)
                samples.append((sample, label))
        
        return samples

class ModelNet40(dict):
    def __init__(self, root, shapenum:10/40, num_points, split, sample_method='random', normalize=True, with_normals=True, random_rot=False):
        for s in split:
            k = s
            if s=='valid':
                s = 'test'
            self[k] = _ModelNet40Dataset(root, s, shapenum, num_points, sample_method=sample_method, normalize=normalize, with_normals=with_normals, random_rot=random_rot)

class MeterModelNet40:
    def __init__(self):
        self.class_pred, self.FourClasspred = 0, 0
        self.num = 0
    def update(self, outputs:torch.Tensor, labels:torch.Tensor):
        with torch.no_grad():
            #assert isinstance(labels, (list, tuple, dict))
            #assert isinstance(outputs, (list, tuple, dict))
            # mainlabel, FourClassifylabel = labels
            # mainscore, FourClassifyscore = outputs # [b, c], [b, 4]
            # mainlabel = mainlabel.to(mainscore).long()
            # FourClassifylabel = FourClassifylabel.to(FourClassifyscore).long()
            # class_pred = F.softmax(mainscore, dim=1).argmax(1) # (b, )
            # FourClass_pred = F.softmax(FourClassifyscore, dim=1).argmax(1) # (b, )
            # self.class_pred += (class_pred==mainlabel).sum()
            # self.FourClasspred += (FourClass_pred==FourClassifylabel).sum()
            
            _, FourClassifylabel = labels
            FourClassifyscore = outputs # [b, c], [b, 4]
            FourClassifylabel = FourClassifylabel.to(FourClassifyscore).long()
            FourClass_pred = F.softmax(FourClassifyscore, dim=1).argmax(1) # (b, )
            
            self.FourClasspred += (FourClass_pred==FourClassifylabel).sum()

            self.num += FourClassifyscore.shape[0] 
            #print(self.pred, self.num)
    def compute(self):
        return {'reflect_acc':self.FourClasspred.item()/self.num}
        #return {'class_acc':self.class_pred.item()/self.num, 'reflect_acc':self.FourClasspred.item()/self.num}

if __name__ == '__main__':
    pc = torch.rand(1000, 3)
    m=50
    start_idx = torch.randint(pc.shape[0], (1, )).long()
    pts = farthest_point_sample(pc, m, start_idx)
    pts2, idx = farthest_point_sample2(pc, m, start_idx)
    print(f'pts = {pts}\npts2 = {pts2}\nidx = {idx}')