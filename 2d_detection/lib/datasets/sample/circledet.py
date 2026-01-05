from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math


class CirCleDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        '''
        确保裁剪在图像内部进行
        '''
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

        # Get the image
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]

        # 图像中心点c
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)

        # 如果keep_res为True，代码计算并保存图像的原始尺寸（加上填充值），并将其作为尺度 s；
        # 如果keep_res为False，代码将图像的最大边长作为尺度 s，并设置输入图像的目标高度和宽度为预设的值。
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        ## 如果是训练模式，则执行其中的数据增强操作
        if self.split == 'train':  
            # Random crop by default（默认随机裁剪）
            if not self.opt.not_rand_crop:   # not_rand_crop = False. 对于store_true的，只有命令行中出现了该变量才会是true.
                # 随机选择一个缩放因子（从0.6到1.4，步长为0.1），并将其应用于尺度 s。这会随机缩放图像的尺度，增强数据的多样性。
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                # 通过 _get_border 函数计算裁剪时在宽度和高度方向上的边界限制，确保裁剪时不会超出图像的边缘。这里 128 是裁剪尺寸的最小值
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                # 随机选择裁剪区域的左上角坐标 c[0] 和 c[1]，确保裁剪区域在图像内部
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

            # 缩放和位移，根据设的值
            else:
                sf = self.opt.scale   # 0.4
                cf = self.opt.shift   # 0.1

                # 缩放
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)

                # 位移
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            # Flip image
            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        # 这三个角度不会改变尺度和中心点
        if self.opt.rotate > 0:  # rotate the image
            if self.opt.rotate == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            if self.opt.rotate == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            if self.opt.rotate == 270:
                img = cv2.rotate(img, cv2.img_rotate_90_counterclockwise)

        # Perform affine transformation
        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])     # 对于输入图像的仿射变换矩阵，inp是input

        # Warp affine
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)

        # Scale RGB pixels
        inp = (inp.astype(np.float32) / 255.)

        # Add color augmentation
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
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

        # Add for circle
        cl = np.zeros((self.max_objs, 1), dtype=np.float32)
        dense_cl = np.zeros((1, output_h, output_w), dtype=np.float32)
        reg_cl = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind_cl = np.zeros((self.max_objs), dtype=np.int64)
        cat_spec_cl = np.zeros((self.max_objs, num_classes * 1), dtype=np.float32)
        cat_spec_clmask = np.zeros((self.max_objs, num_classes * 1), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        # For each object in the annotation
        for k in range(num_objs):
            # Get the annotation
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])

            # Debug print statements
            # print(self.cat_ids)
            # print(ann['category_id'])
            # print(int(self.cat_ids[int(ann['category_id'])]))

            cls_id = int(self.cat_ids[int(ann['category_id'])])

            center_point = ann['circle_center']
            center_radius = ann['circle_radius']

            # If the image was flipped, then flip the annotation
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                center_point[0] = width - center_point[0]

            # If the image was affine transformed, then transform the annotation
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            center_point_aff = affine_transform(center_point, trans_output)
            center_radius_aff = center_radius * trans_output[0][0]
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0 and center_point_aff[0]>0 \
                    and center_point_aff[1]>0 and center_point_aff[0]<output_w\
                    and center_point_aff[1]<output_h:

                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                #
                ct_int = ct.astype(np.int32)
                # # draw_gaussian(hm[cls_id], ct_int, radius)
                # wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                # reg[k] = ct - ct_int

                # cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                # cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                # if self.opt.dense_wh:
                #     draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                # gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                #                ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
                if self.opt.ez_guassian_radius:
                    radius = center_radius_aff
                else:
                    radius = gaussian_radius((math.ceil(center_radius_aff*2), math.ceil(center_radius_aff*2)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                cp = center_point_aff
                cp_int = cp.astype(np.int32)
                draw_gaussian(hm[cls_id], cp_int, radius)
                ind_cl[k] = cp_int[1] * output_w + cp_int[0]
                reg_cl[k] = cp - cp_int
                reg_mask[k] = 1
                cr = center_radius_aff
                cl[k] = 1. * cr
                cat_spec_cl[k, cls_id * 1: cls_id * 1 + 1] = cl[k]
                cat_spec_clmask[k, cls_id * 1: cls_id * 1 + 1] = 1
                if self.opt.filter_boarder:
                    if cp[0] - cr < 0 or cp[0] + cr > output_w:
                        continue
                    if cp[1] - cr < 0 or cp[1] + cr > output_h:
                        continue
                gt_det.append([cp[0], cp[1], cr, 1, cls_id])

                # if ind_cl[0]<0:
                #     aaa = 1
                #
                # print('ind')
                # print(ind[0:10])
                # print('ind_cl')
                # print(ind_cl[0:10])

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind_cl, 'cl': cl}
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg_cl})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 5), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret