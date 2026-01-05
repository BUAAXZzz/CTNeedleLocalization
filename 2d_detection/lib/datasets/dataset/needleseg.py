from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import datasets.eval_protocals.kidpath_circle as kidpath_circle
from datasets.eval_protocals.circle_eval import CIRCLEeval
from pycocotools.cocoeval import COCOeval
from datasets.eval_protocals.keypt_eval import KEYPOINTeval

import numpy as np
import json
import os
import torch.utils.data as data


class NeedleSeg(data.Dataset):
    num_classes = 2
    default_resolution = [512, 512]

    # Default normalization stats (may be overridden below)
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(NeedleSeg, self).__init__()

        # Dataset paths
        self.data_dir = os.path.join(opt.data_dir, 'needleseg')
        self.img_dir = os.path.join(self.data_dir, '{}'.format(split))
        self.annot_path = os.path.join(
            self.data_dir, 'NeedleSeg_{}2025.json').format(split)

        # Basic settings
        self.max_objs = 32
        self.class_name = ['__background__', 'handle', 'tip']
        self._valid_ids = [1, 2]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

        # Color map for visualization (VOC-style)
        self.voc_color = [
            (v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32)
            for v in range(1, self.num_classes + 1)
        ]

        # Random generator for data augmentation
        self._data_rng = np.random.RandomState(123)

        # Eigenvalues and eigenvectors used in color augmentation (if enabled)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        # Normalization stats (ImageNet-style). This overrides the default above.
        self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        opt.default_resolution = self.default_resolution
        self.opt = opt

        print('==> initializing NeedleSeg 2025 {} data.'.format(split))

        # COCO API for bbox-style annotations
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        # Circle API for circle/keypoint evaluation protocol
        self.circle = kidpath_circle.CIRCLE(self.annot_path)
        self.images_circle = self.circle.getImgIds()
        self.num_samples_circle = len(self.images_circle)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        """
        Convert a number to float with 2-decimal formatting.
        """
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        """
        Convert internal bbox results to COCO evaluation JSON format.
        Input bbox format is expected as [x1, y1, x2, y2, score, ...].
        Output bbox format uses [x, y, w, h].
        """
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def convert_eval_circle_format(self, all_circles):
        """
        Convert internal circle results to circle evaluation JSON format.
        Expected circle format: [cx, cy, r, score, ...]
        """
        detections = []
        for image_id in all_circles:
            for cls_ind in all_circles[image_id]:
                if cls_ind - 1 < 0 or cls_ind - 1 >= len(self._valid_ids):
                    # Skip invalid class index
                    continue
                category_id = self._valid_ids[cls_ind - 1]

                for circle in all_circles[image_id][cls_ind]:
                    score = circle[3]
                    circle_out = list(map(self._to_float, circle[0:3]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "score": float("{:.2f}".format(score)),
                        "circle_center": [circle_out[0], circle_out[1]],
                        "circle_radius": circle_out[2]
                    }
                    if len(circle) > 5:
                        extreme_points = list(map(self._to_float, circle[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def convert_eval_keypoint_format(self, all_keypoints):
        """
        Convert internal keypoint results to keypoint evaluation JSON format.
        Expected keypoint format: [cx, cy, r, angle, score, ...]
        """
        detections = []
        for image_id in all_keypoints:
            for cls_ind in all_keypoints[image_id]:
                if cls_ind - 1 < 0 or cls_ind - 1 >= len(self._valid_ids):
                    # Skip invalid class index
                    continue
                category_id = self._valid_ids[cls_ind - 1]

                for kpt in all_keypoints[image_id][cls_ind]:
                    score = kpt[4]
                    kpt_out = list(map(self._to_float, kpt[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "score": float("{:.2f}".format(score)),
                        "keypoint_center": [kpt_out[0], kpt_out[1]],
                        "keypoint_radius": kpt_out[2],
                        "keypoint_angle": kpt_out[3]
                    }
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        """
        Save bbox results to COCO-format JSON file.
        """
        json.dump(self.convert_eval_format(results),
                  open('{}/results_bbox.json'.format(save_dir), 'w'))

    # ----------------------------
    # Keypoint evaluation
    # ----------------------------
    def run_keypoint_eval(self, results, save_dir):
        """
        Run keypoint evaluation with custom KEYPOINTeval protocol.
        """
        self.save_keypts_results(results, save_dir)
        kpt_dets = self.circle.loadRes('{}/results_keypoints.json'.format(save_dir))
        kpt_eval = KEYPOINTeval(self.circle, kpt_dets, "keypoints")
        kpt_eval.evaluate()
        kpt_eval.accumulate()
        kpt_eval.summarize()

    def save_keypts_results(self, results, save_dir):
        """
        Save keypoint results to JSON file.
        """
        json.dump(self.convert_eval_keypoint_format(results),
                  open('{}/results_keypoints.json'.format(save_dir), 'w'))

    # ----------------------------
    # Bounding box evaluation
    # ----------------------------
    def run_eval(self, results, save_dir):
        """
        Run COCO bbox evaluation. Circle results are also saved for convenience.
        """
        self.save_results(results, save_dir)
        self.save_circle_results(results, save_dir)

        coco_dets = self.coco.loadRes('{}/results_bbox.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    # ----------------------------
    # Circle evaluation
    # ----------------------------
    def save_circle_results(self, results, save_dir):
        """
        Save circle results to JSON file.
        """
        json.dump(self.convert_eval_circle_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_circle_eval(self, results, save_dir):
        """
        Run circle evaluation with custom CIRCLEeval protocol.
        """
        self.save_circle_results(results, save_dir)
        circle_dets = self.circle.loadRes('{}/results.json'.format(save_dir))
        circle_eval = CIRCLEeval(self.circle, circle_dets, "circle")
        circle_eval.evaluate()
        circle_eval.accumulate()
        circle_eval.summarize()
