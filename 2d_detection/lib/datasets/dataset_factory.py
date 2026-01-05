from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ctdet import CTDetDataset
from .sample.circledet import CirCleDataset
from .sample.keyptdet import KeyPtDataset

from .dataset.coco import COCO
from .dataset.monuseg import MoNuSeg
from .dataset.needleseg import NeedleSeg

dataset_factory = {
  'coco': COCO,
  'monuseg': MoNuSeg,
  'needleseg': NeedleSeg
}

_sample_factory = {
  'ctdet': CTDetDataset,
  'circledet': CirCleDataset,
  'keyptdet': KeyPtDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset

