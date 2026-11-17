from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.keyptdet import KeyPtDataset

from .dataset.needleseg import NeedleSeg

dataset_factory = {
  'needleseg': NeedleSeg
}

_sample_factory = {
  'keyptdet': KeyPtDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
