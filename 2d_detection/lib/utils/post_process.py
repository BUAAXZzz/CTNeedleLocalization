from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds


def ctdet_post_process(dets, c, s, h, w, num_classes):
    """
    Post-process bbox detections:
      - Transform predicted coordinates from output space back to image space.
      - Group detections by class (1-based class index in the returned dict).

    Args:
        dets (ndarray): shape [B, K, D], expected fields:
                        [x1, y1, x2, y2, score, cls]
        c, s: affine transform parameters for each image in the batch.
        h, w: output size used in transform (note the order in transform_preds uses (w, h)).
        num_classes (int): number of classes.

    Returns:
        list[dict]: length B, each dict maps class_id (1-based) -> list of [x1, y1, x2, y2, score]
    """
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}

        # Transform bbox corners back to image coordinates
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))

        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)
            ], axis=1).tolist()

        ret.append(top_preds)
    return ret


def circledet_post_process(dets, c, s, h, w, num_classes):
    """
    Post-process circle detections:
      - Transform predicted circle centers back to image coordinates.
      - Adjust radius using a global scale factor (s[i] / w).
      - Group detections by class (1-based class index in the returned dict).

    Args:
        dets (ndarray): shape [B, K, D], expected fields (typical):
                        [cx, cy, r, ..., score, cls]
        c, s: affine transform parameters.
        h, w: output size used in transform.
        num_classes (int): number of classes.

    Returns:
        list[dict]: each dict maps class_id (1-based) -> list of [cx, cy, r, ?, score]
                    Note: the exact meaning of the 4th column depends on your decode output.
    """
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}

        # Transform center coordinates back to image space
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))

        # Radius rescaling. Angle (if any) is not rescaled here.
        dets[i, :, 2] = dets[i, :, 2] * (s[i] / w)

        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)
            ], axis=1).tolist()

        ret.append(top_preds)
    return ret


def keyptdet_post_process(dets, c, s, h, w, num_classes):
    """
    Post-process keypoint-style detections (center + radius + angle):
      - Transform predicted center coordinates back to image space.
      - Rescale radius using (s[i] / w).
      - Keep angle unchanged.
      - Group detections by class (1-based class index in the returned dict).

    Expected input dets per row:
        [x, y, r, angle, score, cls]
    Output per row (as you implemented):
        [x, y, r, angle, score, cls?] is rearranged into:
        concatenate(dets[:5], dets[5:6]) which means:
          - first 5 columns: [x, y, r, angle, score]
          - then one more column: dets[5] (cls) casted to float32

    Note:
      Your current concatenation places cls as the last column (float), not score.
      Make sure your downstream evaluation expects this exact format.

    Args:
        dets (ndarray): shape [B, K, 6], [x, y, r, angle, score, cls]
        c, s: affine transform parameters.
        h, w: output size used in transform.
        num_classes (int): number of classes.

    Returns:
        list[dict]: each dict maps class_id (1-based) -> list of [x, y, r, angle, score, cls_float]
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


def multi_pose_post_process(dets, c, s, h, w):
    """
    Post-process multi-pose detections:
      - Transform bbox and keypoints back to image coordinates.
      - Return a list of dicts, each dict uses a dummy class id as key.

    Args:
        dets (ndarray): shape [B, K, 40]
        c, s: affine transform parameters.
        h, w: output size used in transform.

    Returns:
        list[dict]: each dict maps a dummy class id -> list of detections in image coords.
    """
    ret = []
    for i in range(dets.shape[0]):
        bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
        pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))

        top_preds = np.concatenate(
            [bbox.reshape(-1, 4),
             dets[i, :, 4:5],
             pts.reshape(-1, 34)],
            axis=1
        ).astype(np.float32).tolist()

        ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
    return ret
