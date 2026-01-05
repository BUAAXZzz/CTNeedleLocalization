from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .circledet import CircleTrainer
from .keyptdet import KeyPointTrainer

train_factory = {
  'ctdet': CtdetTrainer,
  'circledet': CircleTrainer,
  'keyptdet': KeyPointTrainer
}
