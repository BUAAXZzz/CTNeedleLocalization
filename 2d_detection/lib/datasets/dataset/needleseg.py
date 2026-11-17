from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import datasets.eval_protocals.kidpath_circle as kidpath_circle
from datasets.eval_protocals.keypt_eval import KEYPOINTeval
from utils.pred2d import export_pred2d

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
        # Keep every endpoint, including slices with more than 32 annotations.
        self.max_objs = max(self.max_objs, max(
            (len(self.coco.imgToAnns[image_id]) for image_id in self.images),
            default=0))

        # Circle API for circle/keypoint evaluation protocol
        self.circle = kidpath_circle.CIRCLE(self.annot_path)
        self.images_circle = self.circle.getImgIds()
        self.num_samples_circle = len(self.images_circle)

        print('Loaded {} {} samples'.format(split, self.num_samples))




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

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "score": float(score),
                        "keypoint_center": [float(kpt[0]), float(kpt[1])],
                        "keypoint_radius": float(kpt[2]),
                        "keypoint_angle": float(kpt[3])
                    }
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples


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
        detections = self.convert_eval_keypoint_format(results)
        with open(os.path.join(save_dir, 'results_keypoints.json'), 'w') as f:
            json.dump(detections, f)
        manifest_path = os.path.join(self.data_dir, 'manifest.json')
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            # Export only images present in this inference run, including those
            # whose result lists are empty. Coordinates were restored by post_process.
            images = self.coco.loadImgs([int(i) for i in results])
            export_pred2d(detections, images, os.path.join(save_dir, 'pred2d'),
                          manifest, score_thresh=self.opt.center_thresh)

    # ----------------------------
    # Bounding box evaluation
    # ----------------------------

    # ----------------------------
    # Circle evaluation
    # ----------------------------
