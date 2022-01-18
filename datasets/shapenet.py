import json
import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.open3d_func import random_rotation
#__all__ = ['ShapeNet']

class _ShapeNetDataset(Dataset):
    def __init__(self, root, num_points, split='train', with_random_rot=True, with_normal=True, with_one_hot_shape_id=True,
                 normalize=True, jitter=True):
        assert split in ['train', 'test']
        self.root = root
        self.num_points = num_points
        self.split = split
        self.with_random_rot = with_random_rot
        self.with_normal = with_normal
        self.with_one_hot_shape_id = with_one_hot_shape_id
        self.normalize = normalize
        self.jitter = jitter

        shape_dir_to_shape_id = {}
        with open(os.path.join(self.root, 'synsetoffset2category.txt'), 'r') as f:
            for shape_id, line in enumerate(f):
                shape_name, shape_dir = line.strip().split()
                shape_dir_to_shape_id[shape_dir] = shape_id
        file_paths = []
        if self.split == 'train':
            #split = ['train', 'val']
            split = ['train']
        else:
            split = ['test']
        for s in split:
            with open(os.path.join(self.root, 'train_test_split', f'shuffled_{s}_file_list.json'), 'r') as f:
                file_list = json.load(f)
                for file_path in file_list:
                    _, shape_dir, filename = file_path.split('/')
                    datafile = os.path.join(self.root, shape_dir, filename + '.txt')
                    if os.path.getsize(datafile):
                        try:
                            data = np.loadtxt(datafile).astype(np.float32)
                            file_paths.append((datafile,shape_dir_to_shape_id[shape_dir]))
                            #print(datafile)
                        except:
                            continue
                        
                    else:
                        continue
        #print(file_paths)
        self.file_paths = file_paths
        self.num_shapes = 16
        self.num_classes = 50

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            coords, normal, label, shape_id = self.cache[index]
        else:
            file_path, shape_id = self.file_paths[index]
            try:
                data = np.loadtxt(file_path).astype(np.float32)
            except:
                print(file_path)
                sys.exit(0)
            coords = data[:, :3] #(n, 3)
            if self.normalize:
                coords = self.normalize_point_cloud(coords)
            normal = data[:, 3:6] #(n, 3)
            label = data[:, -1].astype(np.int64) #(n,)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (coords, normal, label, shape_id)

        choice = np.random.choice(label.shape[0], self.num_points, replace=True)
        coords = coords[choice, :]
        normal = normal[choice, :]
        if self.with_random_rot:
            T, coords, normal = random_rotation(coords, normal, max_degree=360, max_amp=3)
            #print(T)
        coords = coords.transpose()
        normal = normal.transpose()
        if self.jitter:
            coords = self.jitter_point_cloud(coords)
        if self.with_normal:
            if self.with_one_hot_shape_id:
                shape_one_hot = np.zeros((self.num_shapes, self.num_points), dtype=np.float32)
                shape_one_hot[shape_id, :] = 1.0
                point_set = np.concatenate([coords, normal, shape_one_hot])
            else:
                point_set = np.concatenate([coords, normal])
        else:
            if self.with_one_hot_shape_id:
                shape_one_hot = np.zeros((self.num_shapes, self.num_points), dtype=np.float32)
                shape_one_hot[shape_id, :] = 1.0
                point_set = np.concatenate([coords, shape_one_hot])
            else:
                point_set = coords
        return point_set, label[choice].transpose()

    def __len__(self):
        return len(self.file_paths)
        #return 128

    @staticmethod
    def normalize_point_cloud(points):
        centroid = np.mean(points, axis=0)
        points = points - centroid
        return points / np.max(np.linalg.norm(points, axis=1))

    @staticmethod
    def jitter_point_cloud(points, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              3xN array, original batch of point clouds
            Return:
              3xN array, jittered batch of point clouds
        """
        assert (clip > 0)
        return np.clip(sigma * np.random.randn(*points.shape), -1 * clip, clip).astype(np.float32) + points
    
class ShapeNet(dict):
    def __init__(self, root, num_points, split=None, with_random_rot=True,with_normal=True, with_one_hot_shape_id=True,
                 normalize=True, jitter=True):
        super().__init__()
        if split is None:
            split = ['train', 'valid', 'test']
        elif not isinstance(split, (list, tuple)):
            split = [split]
        for s in split:
            k = s
            if s=='valid':
                s = 'test'
            self[k] = _ShapeNetDataset(root=root, num_points=num_points, split=s,with_random_rot=with_random_rot,
                                       with_normal=with_normal, with_one_hot_shape_id=with_one_hot_shape_id,
                                       normalize=normalize, jitter=jitter if s == 'train' else False)

default_shape_name_to_part_classes = {
    'Airplane': [0, 1, 2, 3],
    'Bag': [4, 5],
    'Cap': [6, 7],
    'Car': [8, 9, 10, 11],
    'Chair': [12, 13, 14, 15],
    'Earphone': [16, 17, 18],
    'Guitar': [19, 20, 21],
    'Knife': [22, 23],
    'Lamp': [24, 25, 26, 27],
    'Laptop': [28, 29],
    'Motorbike': [30, 31, 32, 33, 34, 35],
    'Mug': [36, 37],
    'Pistol': [38, 39, 40],
    'Rocket': [41, 42, 43],
    'Skateboard': [44, 45, 46],
    'Table': [47, 48, 49],
}


class MeterShapeNet:
    def __init__(self, shape_name_to_part_classes=None):
        super().__init__()
        self.shape_name_to_part_classes = default_shape_name_to_part_classes if shape_name_to_part_classes is None \
            else shape_name_to_part_classes
        part_class_to_shape_part_classes = []
        for shape_name, shape_part_classes in self.shape_name_to_part_classes.items():
            start_class, end_class = shape_part_classes[0], shape_part_classes[-1] + 1
            for _ in range(start_class, end_class):
                part_class_to_shape_part_classes.append((start_class, end_class))
        # (50,)
        self.part_class_to_shape_part_classes = part_class_to_shape_part_classes
        self.reset()

    def reset(self):
        self.iou_sum = 0
        self.shape_count = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        # outputs: B x num_classes x num_points, targets: B x num_points
        #print(targets.shape, outputs.size(0))
        for b in range(outputs.size(0)):
            # 第一个的part class可以反映出shape class
            #print(b, targets[b, 0])
            start_class, end_class = self.part_class_to_shape_part_classes[targets[b, 0].item()]
            #  只关心本shape的part分类对不对
            prediction = torch.argmax(outputs[b, start_class:end_class, :], dim=0) + start_class
            target = targets[b, :]
            iou = 0.0
            for i in range(start_class, end_class):
                itarget = (target == i)
                iprediction = (prediction == i)
                union = torch.sum(itarget | iprediction).item()
                intersection = torch.sum(itarget & iprediction).item()
                if union == 0:
                    iou += 1.0
                else:
                    iou += intersection / union
            iou /= (end_class - start_class)
            self.iou_sum += iou
            self.shape_count += 1

    def compute(self):
        return self.iou_sum / self.shape_count

if __name__ == '__main__':
    points = np.eye(3)
    points, T = _ShapeNetDataset.random_rotation(points, max_degree=360, max_amp=3)
