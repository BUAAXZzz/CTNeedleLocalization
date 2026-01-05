from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # ----------------------------------------------------------------------
        # Basic experiment settings
        # ----------------------------------------------------------------------
        self.parser.add_argument(
            '--task', default='keyptdet',
            help='ctdet | circledet | keyptdet'
        )
        self.parser.add_argument(
            '--dataset', default='needleseg',
            help='coco | monuseg | needleseg'
        )
        self.parser.add_argument('--exp_id', default='latest')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--ontestdata', action='store_true')

        # Optional runtime filters / options
        self.parser.add_argument('--filter_boarder', action='store_true')
        self.parser.add_argument('--ez_guassian_radius', action='store_true')

        # Debug / visualization
        self.parser.add_argument(
            '--debug', type=int, default=0,
            help='level of visualization. '
                 '1: only show the final detection results; '
                 '2: show the network output features; '
                 '3: use matplotlib to display (useful with notebooks); '
                 '4: save all visualizations to disk'
        )

        # Demo mode
        self.parser.add_argument(
            '--demo', default='',
            help='path to image / image folders / video, or "webcam"'
        )
        self.parser.add_argument(
            '--demo_dir', default='',
            help='output path for demo results'
        )

        # Resume / load model
        self.parser.add_argument(
            '--load_model', default='',
            help='path to pretrained model'
        )
        self.parser.add_argument(
            '--resume', action='store_true',
            help='resume an experiment. Reload optimizer states and set '
                 'load_model to model_last.pth in the exp dir if load_model is empty.'
        )

        # Loss options
        self.parser.add_argument('--mask_focal_loss', action='store_true', help='')
        self.parser.add_argument(
            '--test_val_name', default='',
            help='testing and validation study name'
        )

        # ----------------------------------------------------------------------
        # System settings
        # ----------------------------------------------------------------------
        self.parser.add_argument(
            '--gpus', default='1',
            help='-1 for CPU, use comma for multiple gpus'
        )
        self.parser.add_argument(
            '--num_workers', type=int, default=4,
            help='dataloader threads. Use 0 for debugging.'
        )
        self.parser.add_argument(
            '--not_cuda_benchmark', action='store_true',
            help='disable when input size is not fixed'
        )
        self.parser.add_argument(
            '--seed', type=int, default=317,
            help='random seed (from CornerNet)'
        )

        # ----------------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------------
        self.parser.add_argument(
            '--print_iter', type=int, default=0,
            help='disable progress bar and print to screen'
        )
        self.parser.add_argument(
            '--hide_data_time', action='store_true',
            help='not display time during training'
        )
        self.parser.add_argument(
            '--save_all', action='store_true',
            help='save model to disk every 5 epochs'
        )
        self.parser.add_argument(
            '--metric', default='loss',
            help='main metric to save best model'
        )
        self.parser.add_argument(
            '--vis_thresh', type=float, default=0.2,
            help='visualization threshold'
        )
        self.parser.add_argument(
            '--debugger_theme', default='white',
            choices=['white', 'black']
        )

        # ----------------------------------------------------------------------
        # Model settings
        # ----------------------------------------------------------------------
        self.parser.add_argument(
            '--arch', default='hrnet',
            help='model architecture. Currently tested: '
                 'hrnet'
        )
        self.parser.add_argument(
            '--head_conv', type=int, default=-1,
            help='conv layer channels for output head. '
                 '0 for no conv layer; '
                 '-1 for default setting: 64 for resnets and 256 for dla'
        )
        self.parser.add_argument(
            '--down_ratio', type=int, default=4,
            help='output stride. Currently only supports 4'
        )

        # ----------------------------------------------------------------------
        # Input settings
        # ----------------------------------------------------------------------
        self.parser.add_argument(
            '--input_res', type=int, default=-1,
            help='input height and width. -1 for default from dataset. '
                 'Will be overridden by input_h | input_w'
        )
        self.parser.add_argument('--input_h', type=int, default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1,
                                 help='input width. -1 for default from dataset.')

        # ----------------------------------------------------------------------
        # Training settings
        # ----------------------------------------------------------------------
        self.parser.add_argument(
            '--lr', type=float, default=1.25e-4,
            help='learning rate for batch size 32'
        )
        self.parser.add_argument(
            '--lr_step', type=str, default='90,120',
            help='drop learning rate by 10 at these epochs'
        )
        self.parser.add_argument(
            '--num_epochs', type=int, default=100,
            help='total training epochs'
        )
        self.parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        self.parser.add_argument(
            '--master_batch_size', type=int, default=-1,
            help='batch size on the master gpu'
        )
        self.parser.add_argument(
            '--num_iters', type=int, default=-1,
            help='default: #samples / batch_size'
        )
        self.parser.add_argument(
            '--val_intervals', type=int, default=1,
            help='number of epochs to run validation'
        )
        self.parser.add_argument(
            '--trainval', action='store_true',
            help='include validation in training and test on test set'
        )

        # ----------------------------------------------------------------------
        # Testing settings
        # ----------------------------------------------------------------------
        self.parser.add_argument('--flip_test', action='store_true',
                                 help='flip data augmentation during testing')
        self.parser.add_argument('--test_scales', type=str, default='1',
                                 help='multi-scale testing')
        self.parser.add_argument('--nms', action='store_true',
                                 help='run NMS in testing')
        self.parser.add_argument('--K', type=int, default=32,
                                 help='max number of output objects')
        self.parser.add_argument('--not_prefetch_test', action='store_true',
                                 help='not use parallel data pre-processing')
        self.parser.add_argument('--fix_res', action='store_true',
                                 help='fix testing resolution or keep original resolution')
        self.parser.add_argument('--keep_res', action='store_true',
                                 help='keep original resolution during validation')
        self.parser.add_argument('--rotate_reproduce', type=float, default=0,
                                 help='rotate 90/180/270 to check consistency')
        self.parser.add_argument('--lv', type=int, default=2,
                                 help='level of simg')

        # ----------------------------------------------------------------------
        # Dataset augmentation settings
        # ----------------------------------------------------------------------
        self.parser.add_argument('--not_rand_crop', action='store_true',
                                 help='disable random crop augmentation from CornerNet')
        self.parser.add_argument('--shift', type=float, default=0.1,
                                 help='shift augmentation factor (used when not using random crop)')
        self.parser.add_argument('--scale', type=float, default=0.4,
                                 help='scale augmentation factor (used when not using random crop)')
        self.parser.add_argument('--rotate', type=float, default=0,
                                 help='rotation augmentation factor (used when not using random crop)')
        self.parser.add_argument('--flip', type=float, default=0.5,
                                 help='probability of applying flip augmentation')
        self.parser.add_argument('--no_color_aug', action='store_true',
                                 help='disable color augmentation from CornerNet')

        # ----------------------------------------------------------------------
        # Loss settings
        # ----------------------------------------------------------------------
        self.parser.add_argument('--mse_loss', action='store_true',
                                 help='use MSE loss instead of focal loss for heatmaps')
        self.parser.add_argument('--reg_loss', default='l1',
                                 help='regression loss: sl1 | l1 | l2')
        self.parser.add_argument('--hm_weight', type=float, default=2,
                                 help='loss weight for keypoint heatmaps')
        self.parser.add_argument('--off_weight', type=float, default=1,
                                 help='loss weight for local offsets')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='loss weight for box size / radius-like term')
        self.parser.add_argument('--angle_weight', type=float, default=1,
                                 help='loss weight for angle prediction')

        # Task-specific threshold
        self.parser.add_argument('--center_thresh', type=float, default=0.1,
                                 help='threshold for center heatmap')

        # ctdet options
        self.parser.add_argument('--norm_wh', action='store_true',
                                 help='L1(ŷ / y, 1) or L1(ŷ, y)')
        self.parser.add_argument('--dense_wh', action='store_true',
                                 help='apply weighted regression near center, or only at center')
        self.parser.add_argument('--cat_spec_wh', action='store_true',
                                 help='category-specific size regression')
        self.parser.add_argument('--not_reg_offset', action='store_true',
                                 help='do not regress local offset')

        # ----------------------------------------------------------------------
        # Oracle evaluation options (for debugging / ablation)
        # ----------------------------------------------------------------------
        self.parser.add_argument('--eval_oracle_hm', action='store_true',
                                 help='use ground truth center heatmap')
        self.parser.add_argument('--eval_oracle_wh', action='store_true',
                                 help='use ground truth bbox size / radius')
        self.parser.add_argument('--eval_oracle_offset', action='store_true',
                                 help='use ground truth local offset')
        self.parser.add_argument('--eval_oracle_angle', action='store_true',
                                 help='use ground truth object angle')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        # ------------------------------------------------------------------
        # Optional: quick override for testing (kept as a commented template)
        # ------------------------------------------------------------------
        """
        # Example overrides for quick testing:
        opt.batch_size = 4
        opt.dataset = 'needleseg'
        opt.load_model = './exp/keyptdet/1209_bt40_notanh/model_last.pth'
        opt.test = True
        opt.debug = 4
        opt.vis_thresh = 0.2
        opt.center_thresh = 0.2
        """

        # Parse and normalize gpu settings
        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]

        # Parse lr steps and test scales
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

        # Resolution settings
        opt.fix_res = not opt.keep_res
        print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')

        # Offset regression flag
        opt.reg_offset = not opt.not_reg_offset

        # Default head conv channels
        if opt.head_conv == -1:
            opt.head_conv = 256 if 'dla' in opt.arch else 64

        # Padding and stacks for specific architectures
        opt.pad = 127 if 'hourglass' in opt.arch else 31
        opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

        # If trainval is enabled, effectively disable validation
        if opt.trainval:
            opt.val_intervals = 100000000

        # Debug mode: force single GPU, single worker, batch size 1
        if opt.debug > 0:
            opt.num_workers = 0
            opt.batch_size = 1
            opt.gpus = [opt.gpus[0]]
            opt.master_batch_size = -1

        # Chunk sizes for multi-GPU training
        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)

        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)

        print('training chunk_sizes:', opt.chunk_sizes)

        # Directory structure
        opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        opt.data_dir = os.path.join(opt.root_dir, 'data')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug_bt40')

        print('The output will be saved to', opt.save_dir)

        # Auto-resume behavior
        if opt.resume and opt.load_model == '':
            model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') else opt.save_dir
            opt.load_model = os.path.join(model_path, 'model_last.pth')

        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        """
        Update input/output resolution, normalization statistics, class number,
        and set the output heads according to the task.
        """
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes

        # input_h/w priority: opt.input_h/w > opt.input_res > dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w

        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        # Define output heads for each task
        if opt.task == 'ctdet':
            opt.heads = {
                'hm': opt.num_classes,
                'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes
            }
            if opt.reg_offset:
                opt.heads.update({'reg': 2})

        elif opt.task == 'circledet':
            opt.heads = {
                'hm': opt.num_classes,
                'cl': 1 if not opt.cat_spec_wh else 1 * opt.num_classes
            }
            if opt.reg_offset:
                opt.heads.update({'reg': 2})

        elif opt.task == 'keyptdet':
            opt.heads = {
                'hm': opt.num_classes,
                'cl': 1 if not opt.cat_spec_wh else 1 * opt.num_classes,
                'angle': 1
            }
            if opt.reg_offset:
                opt.heads.update({'reg': 2})

        else:
            assert 0, 'task not defined!'

        print('heads', opt.heads)
        return opt

    def init(self, args=''):
        """
        Initialize with task-based default dataset settings, then override via CLI args.
        """
        default_dataset_info = {
            'ctdet': {
                'default_resolution': [512, 512],
                'num_classes': 80,
                'mean': [0.408, 0.447, 0.470],
                'std': [0.289, 0.274, 0.278],
                'dataset': 'coco'
            },
            'circledet': {
                'default_resolution': [512, 512],
                'num_classes': 1,
                'mean': [0.408, 0.447, 0.470],
                'std': [0.289, 0.274, 0.278],
                'dataset': 'monuseg'
            },
            'keyptdet': {
                'default_resolution': [512, 512],
                'num_classes': 2,
                'mean': [0.408, 0.447, 0.470],
                'std': [0.289, 0.274, 0.278],
                'dataset': 'needleseg'
            }
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        opt = self.parse(args)
        dataset = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt
