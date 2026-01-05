from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, FocalLoss_mask
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, AngleCosineLoss
from models.decode import keyptdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import keyptdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer


class KeyPointLoss(torch.nn.Module):
  def __init__(self, opt):
    super(KeyPointLoss, self).__init__()

    # Heatmap loss for center prediction
    # If you want masked focal loss, switch to FocalLoss_mask().
    self.crit = FocalLoss()
    # self.crit = FocalLoss_mask()

    # Regression loss type for offset / radius / angle related heads
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None

    # Width-height loss (kept for compatibility; may be unused in keypoint mode)
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg

    # Angle loss: cosine-based loss between predicted angle representation and GT
    self.crit_angle = AngleCosineLoss()

    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt

    # Loss components:
    # hm_loss: center heatmap loss
    # cl_loss: radius (or size-like) regression loss, using output['cl']
    # off_loss: center offset regression loss, using output['reg']
    # angle_loss: angle regression loss, using output['angle']
    hm_loss, cl_loss, off_loss, angle_loss = 0, 0, 0, 0

    for s in range(opt.num_stacks):
      output = outputs[s]

      # Apply sigmoid to heatmap when not using MSE loss
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])

      # Oracle evaluation options (for ablation / debugging)
      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']

      if opt.eval_oracle_wh:
        output['wh'] = torch.from_numpy(gen_oracle_map(
          batch['wh'].detach().cpu().numpy(),
          batch['ind'].detach().cpu().numpy(),
          output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)

      if opt.eval_oracle_offset:
        output['reg'] = torch.from_numpy(gen_oracle_map(
          batch['reg'].detach().cpu().numpy(),
          batch['ind'].detach().cpu().numpy(),
          output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

      if opt.eval_oracle_angle:
        output['angle'] = torch.from_numpy(gen_oracle_map(
          batch['angle'].detach().cpu().numpy(),
          batch['ind'].detach().cpu().numpy(),
          output['angle'].shape[3], output['angle'].shape[2])).to(opt.device)

      # Heatmap loss
      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks

      # Radius/size regression (here it uses output['cl'] and batch['cl'])
      if opt.wh_weight > 0:
        if opt.dense_wh:
          mask_weight = batch['dense_wh_mask'].sum() + 1e-4
          cl_loss += (
            self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                         batch['dense_wh'] * batch['dense_wh_mask']) /
            mask_weight) / opt.num_stacks
        elif opt.cat_spec_wh:
          cl_loss += self.crit_wh(
            output['wh'], batch['cat_spec_mask'],
            batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
        else:
          # In this keypoint setting, 'cl' typically represents radius
          cl_loss += self.crit_reg(
            output['cl'], batch['reg_mask'],
            batch['ind'], batch['cl']) / opt.num_stacks

      # Angle regression loss
      if opt.angle_weight > 0:
        # Alternative: using regression loss directly
        # angle_loss += self.crit_reg(
        #     output['angle'], batch['reg_mask'],
        #     batch['ind'], batch['angle']) / opt.num_stacks
        angle_loss += self.crit_angle(
          output['angle'], batch['reg_mask'],
          batch['ind'], batch['angle']) / opt.num_stacks

      # Offset regression loss
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(
          output['reg'], batch['reg_mask'],
          batch['ind'], batch['reg']) / opt.num_stacks

    # Weighted total loss
    loss = opt.hm_weight * hm_loss + opt.wh_weight * cl_loss + \
           opt.off_weight * off_loss + opt.angle_weight * angle_loss

    loss_stats = {
      'loss': loss,
      'hm_loss': hm_loss,
      'cl_loss': cl_loss,
      'off_loss': off_loss,
      'angle_loss': angle_loss
    }
    return loss, loss_stats


class KeyPointTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(KeyPointTrainer, self).__init__(opt, model, optimizer=optimizer)

  def _get_losses(self, opt):
    # Expose loss stats to logger
    loss_states = ['loss', 'hm_loss', 'cl_loss', 'off_loss', 'angle_loss']
    loss = KeyPointLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt

    # Optional regression head for sub-pixel offset
    reg = output['reg'] if opt.reg_offset else None

    # Decode network outputs to detections
    # dets is typically shaped [B, K, D] where D includes center, radius, score, class, angle, etc.
    dets = keyptdet_decode(
      output['hm'], output['cl'], output['angle'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K
    )

    # Optional border filtering: suppress detections whose circle exceeds feature map boundaries
    if opt.filter_boarder:
      output_h = self.opt.default_resolution[0] // self.opt.down_ratio  # hard coded
      output_w = self.opt.default_resolution[1] // self.opt.down_ratio  # hard coded
      for i in range(dets.shape[1]):
        cp = [0, 0]
        cp[0] = dets[0, i, 0]
        cp[1] = dets[0, i, 1]
        cr = dets[0, i, 2]
        if cp[0] - cr < 0 or cp[0] + cr > output_w:
          dets[0, i, 3] = 0
          continue
        if cp[1] - cr < 0 or cp[1] + cr > output_h:
          dets[0, i, 3] = 0
          continue

    # Move to CPU and scale back to input resolution
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :3] *= opt.down_ratio

    # Ground truth detections for visualization
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :3] *= opt.down_ratio

    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme
      )

      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)

      # Denormalize using ImageNet mean/std, then stretch to 0-255 for visualization
      # Note: the min-max normalization is used here to improve contrast, not to preserve absolute intensity.
      mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
      std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
      img = img * std + mean
      img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())

      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')

      # Draw predicted detections
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_arrow(
            dets[i, k, :4], dets[i, k, -1],
            dets[i, k, 4], img_id='out_pred'
          )

      debugger.add_img(img, img_id='out_gt')

      # Draw ground truth detections
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_arrow(
            dets_gt[i, k, :4], dets_gt[i, k, -1],
            dets_gt[i, k, 4], img_id='out_gt'
          )

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    # reg_offset is enabled in your setting; output['reg'] is usually [B, 2, H, W]
    reg = output['reg'] if self.opt.reg_offset else None

    # Decode predictions into top-K detections
    dets = keyptdet_decode(
      output['hm'], output['cl'], output['angle'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K
    )

    opt = self.opt

    # Optional border filtering in feature map coordinate space
    if opt.filter_boarder:
      output_h = self.opt.default_resolution[0] // self.opt.down_ratio  # hard coded
      output_w = self.opt.default_resolution[1] // self.opt.down_ratio  # hard coded
      for i in range(dets.shape[1]):
        cp = [0, 0]
        cp[0] = dets[0, i, 0]
        cp[1] = dets[0, i, 1]
        cr = dets[0, i, 2]
        if cp[0] - cr < 0 or cp[0] + cr > output_w:
          dets[0, i, 3] = 0
          continue
        if cp[1] - cr < 0 or cp[1] + cr > output_h:
          dets[0, i, 3] = 0
          continue

    # Convert to numpy and run post-processing to map back to original image coordinates
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = keyptdet_post_process(
      dets.copy(),
      batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1]
    )

    # Store results indexed by image id
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
