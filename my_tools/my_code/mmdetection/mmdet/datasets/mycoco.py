from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class MyUOD(CocoDataset):

    CLASSES = ('holothurian', 'echinus', 'scallop', 'starfish')
