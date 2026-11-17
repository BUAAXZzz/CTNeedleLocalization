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

        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]

        c = np.array([width / 2.0, height / 2.0], dtype=np.float32)

        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:

            s = max(height, width) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False

        if self.split == 'train':

            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, width)
                h_border = self._get_border(128, height)
                c[0] = np.random.randint(low=w_border, high=width - w_border)
                c[1] = np.random.randint(low=h_border, high=height - h_border)
            else:

                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h),
                             flags=cv2.INTER_LINEAR)

        inp = (inp.astype(np.float32) / 255.0)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes

        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)

        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        cl = np.zeros((self.max_objs, 1), dtype=np.float32)
        dense_cl = np.zeros((1, output_h, output_w), dtype=np.float32)
        reg_cl = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind_cl = np.zeros((self.max_objs), dtype=np.int64)
        cat_spec_cl = np.zeros((self.max_objs, num_classes * 1), dtype=np.float32)
        cat_spec_clmask = np.zeros((self.max_objs, num_classes * 1), dtype=np.uint8)

        angle_kp = np.zeros((self.max_objs, 1), dtype=np.float32)
        angle_mask = np.zeros((self.max_objs), dtype=np.uint8)
        reg_kp = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind_kp = np.zeros((self.max_objs), dtype=np.int64)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

        gt_det = []

        for k in range(num_objs):
            ann = anns[k]

            direction = list(ann['direction'])
            angle_gt = math.atan2(-direction[1], direction[0])

            cls_id = int(self.cat_ids[int(ann['category_id'])])

            center_point = list(ann['circle_center'])
            center_radius = ann['circle_radius']

            if flipped:
                center_point[0] = width - center_point[0] - 1
                direction[0] = -direction[0]
                angle_gt = math.atan2(-direction[1], direction[0])

            cp = affine_transform(center_point, trans_output)

            cr = float(center_radius * trans_output[0][0])

            if not (cp[0] > 0 and cp[1] > 0 and cp[0] < output_w and cp[1] < output_h):
                continue

            cp_int = cp.astype(np.int32)

            if self.opt.filter_boarder:
                if (cp[0] - cr < 0) or (cp[0] + cr > output_w):
                    continue
                if (cp[1] - cr < 0) or (cp[1] + cr > output_h):
                    continue

            if self.opt.ez_guassian_radius:
                radius = cr
            else:
                radius = gaussian_radius((math.ceil(cr * 2), math.ceil(cr * 2)))
            radius = max(0, int(radius))
            radius = self.opt.hm_gauss if self.opt.mse_loss else radius

            draw_gaussian(hm[cls_id], cp_int, radius)

            ind_cl[k] = cp_int[1] * output_w + cp_int[0]
            reg_cl[k] = cp - cp_int

            ind_kp[k] = cp_int[1] * output_w + cp_int[0]
            reg_kp[k] = cp - cp_int
            reg_mask[k] = 1

            cl[k] = cr
            cat_spec_cl[k, cls_id * 1: cls_id * 1 + 1] = cl[k]
            cat_spec_clmask[k, cls_id * 1: cls_id * 1 + 1] = 1

            angle_kp[k] = angle_gt
            angle_mask[k] = int(ann.get('angle_valid', True))

            gt_det.append([cp[0], cp[1], cr, angle_gt, 1, cls_id])

        ret = {
            'input': inp,
            'hm': hm,
            'reg_mask': reg_mask,
            'ind': ind_kp,
            'cl': cl,
            'angle': angle_kp
        }
        ret['angle_mask'] = angle_mask

        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})

        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})

        if self.opt.reg_offset:
            ret.update({'reg': reg_kp})

        if self.opt.debug > 0 or self.split != 'train':
            if len(gt_det) > 0:
                gt_det = np.array(gt_det, dtype=np.float32)
            else:
                gt_det = np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta

        return ret
