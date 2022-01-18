
from utils.config import Config, configs
from datasets.shapenet import ShapeNet, MeterShapeNet

configs.dataset = Config(ShapeNet, split=['train', 'valid', 'test'])
configs.dataset.root = "/media/zebai/T7/Datasets/shape_data"
configs.dataset.num_points = 1024

configs.evaluate = Config()
configs.evaluate.meters = Config()
configs.evaluate.meters['eval-iou_{}'] = Config(MeterShapeNet)
configs.evaluate.fn = None