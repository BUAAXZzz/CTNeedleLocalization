from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import cv2
import os
import math

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg


class KeyPtDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        """
        Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2].
        """
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        """
        Dynamically shrink the border size so that the random center sampling
        will not collapse when the image is small.
        """
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        # ------------------------------------------------------------
        # 1) Load image and annotations
        # ------------------------------------------------------------
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]

        # ------------------------------------------------------------
        # 2) Data augmentation setup (center c and scale s)
        # ------------------------------------------------------------
        # c: image center
        # s: scaling factor that defines the crop region before warping
        c = np.array([width / 2.0, height / 2.0], dtype=np.float32)

        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            # Use a square crop based on the longer side
            s = max(height, width) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False

        if self.split == 'train':
            # Random crop is used by default unless not_rand_crop is enabled
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, width)
                h_border = self._get_border(128, height)
                c[0] = np.random.randint(low=w_border, high=width - w_border)
                c[1] = np.random.randint(low=h_border, high=height - h_border)
            else:
                # Otherwise use scale + shift augmentation
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            # Horizontal flip augmentation
            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        # Note: color augmentation is omitted here as in your original snippet.

        # ------------------------------------------------------------
        # 3) Affine transform (input and output)
        # ------------------------------------------------------------
        # Transform the (possibly random-cropped) region to the fixed network input size
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h),
                             flags=cv2.INTER_LINEAR)

        # Normalize and convert HWC to CHW
        inp = (inp.astype(np.float32) / 255.0)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        # Output resolution after downsampling
        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes

        # Transform annotations into output space
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        # ------------------------------------------------------------
        # 4) Initialize target tensors
        # ------------------------------------------------------------
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)

        # Box-related targets (kept for compatibility, even if not used downstream)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        # Circle-related targets
        cl = np.zeros((self.max_objs, 1), dtype=np.float32)
        dense_cl = np.zeros((1, output_h, output_w), dtype=np.float32)
        reg_cl = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind_cl = np.zeros((self.max_objs), dtype=np.int64)
        cat_spec_cl = np.zeros((self.max_objs, num_classes * 1), dtype=np.float32)
        cat_spec_clmask = np.zeros((self.max_objs, num_classes * 1), dtype=np.uint8)

        # Keypoint-related targets
        angle_kp = np.zeros((self.max_objs, 1), dtype=np.float32)
        reg_kp = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind_kp = np.zeros((self.max_objs), dtype=np.int64)

        # Choose Gaussian drawing function
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

        gt_det = []

        # ------------------------------------------------------------
        # 5) Build supervision for each object
        # ------------------------------------------------------------
        for k in range(num_objs):
            ann = anns[k]

            # Direction vector and angle definition
            direction = ann['direction']
            angle_gt = math.atan2(-direction[1], direction[0])

            # Map category id to contiguous training id
            cls_id = int(self.cat_ids[int(ann['category_id'])])

            # Circle center and radius in original image space
            center_point = ann['circle_center']
            center_radius = ann['circle_radius']

            # Apply flip to annotation if needed
            if flipped:
                center_point[0] = width - center_point[0]
                direction[0] = -direction[0]
                angle_gt = math.atan2(-direction[1], direction[0])

            # Transform center point into output coordinate space
            cp = affine_transform(center_point, trans_output)

            # Approximate radius scaling by affine scale (no rotation case)
            # This assumes trans_output is isotropic scaling along x.
            cr = float(center_radius * trans_output[0][0])

            # Check if center is inside the valid output map
            if not (cp[0] > 0 and cp[1] > 0 and cp[0] < output_w and cp[1] < output_h):
                continue

            cp_int = cp.astype(np.int32)

            # Border filtering: reject objects whose circle would exceed the map boundary
            # Important: do this before writing any supervision to avoid label pollution.
            if self.opt.filter_boarder:
                if (cp[0] - cr < 0) or (cp[0] + cr > output_w):
                    continue
                if (cp[1] - cr < 0) or (cp[1] + cr > output_h):
                    continue

            # Compute Gaussian radius
            if self.opt.ez_guassian_radius:
                radius = cr
            else:
                radius = gaussian_radius((math.ceil(cr * 2), math.ceil(cr * 2)))
            radius = max(0, int(radius))
            radius = self.opt.hm_gauss if self.opt.mse_loss else radius

            # Draw center heatmap
            draw_gaussian(hm[cls_id], cp_int, radius)

            # Write regression supervision after all filters are passed
            ind_cl[k] = cp_int[1] * output_w + cp_int[0]
            reg_cl[k] = cp - cp_int

            ind_kp[k] = cp_int[1] * output_w + cp_int[0]
            reg_kp[k] = cp - cp_int
            reg_mask[k] = 1

            cl[k] = cr
            cat_spec_cl[k, cls_id * 1: cls_id * 1 + 1] = cl[k]
            cat_spec_clmask[k, cls_id * 1: cls_id * 1 + 1] = 1

            angle_kp[k] = angle_gt

            # For debug / visualization
            gt_det.append([cp[0], cp[1], cr, angle_gt, 1, cls_id])

        # ------------------------------------------------------------
        # 6) Pack return dictionary
        # ------------------------------------------------------------
        ret = {
            'input': inp,
            'hm': hm,
            'reg_mask': reg_mask,
            'ind': ind_kp,
            'cl': cl,
            'angle': angle_kp
        }

        # Dense width-height branch (kept for compatibility with other heads)
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            # In some codebases, 'wh' would be removed here, but it is not in ret.
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            # In some codebases, 'wh' would be removed here, but it is not in ret.

        # Offset regression
        if self.opt.reg_offset:
            ret.update({'reg': reg_kp})

        # ------------------------------------------------------------
        # 7) Meta info for debugging or evaluation
        # ------------------------------------------------------------
        if self.opt.debug > 0 or self.split != 'train':
            if len(gt_det) > 0:
                gt_det = np.array(gt_det, dtype=np.float32)
            else:
                gt_det = np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta

        return ret
