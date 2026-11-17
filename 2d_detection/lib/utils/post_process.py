from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds






def keyptdet_post_process(dets, c, s, h, w, num_classes):
    """Restore [x, y, radius, angle, score, class] to original slice coordinates.

    Return one dictionary per image, keyed by one-based category IDs.
    """
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}

        # Transform center coordinates back to image coordinates
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))

        # Radius rescaling. Angle is kept unchanged.
        dets[i, :, 2] = dets[i, :, 2] * (s[i] / w)

        # Group by class
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :5].astype(np.float32),   # [x, y, r, angle, score]
                dets[i, inds, 5:6].astype(np.float32)   # [cls] as float
            ], axis=1).tolist()

        ret.append(top_preds)
    return ret
