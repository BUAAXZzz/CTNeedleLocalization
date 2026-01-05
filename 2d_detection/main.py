from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import copy
import torch
import numpy as np
import math
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
import time


def main(opt):
    # Set random seed for reproducibility
    torch.manual_seed(opt.seed)

    # Enable cuDNN benchmark for faster training when applicable
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    # Initialize dataset class
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    # Initialize logger
    logger = Logger(opt)

    # Set visible GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step
        )

    # Initialize trainer
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data loaders...')

    # Validation data loader
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # Test or validation loader depending on flag
    if opt.ontestdata:
        test_loader = torch.utils.data.DataLoader(
            Dataset(opt, 'test'),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            Dataset(opt, 'val'),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

    # ----------------------------
    # Testing stage
    # ----------------------------
    if opt.test:
        start_time = time.time()

        _, preds = trainer.val(0, test_loader)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Executed in: {execution_time:.2f} seconds")

        # Run task-specific evaluation
        if opt.task == 'keyptdet':
            test_loader.dataset.run_keypoint_eval(preds, opt.save_dir)
        else:
            test_loader.dataset.run_eval(preds, opt.save_dir)
        return

    # Training data loader
    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # ----------------------------
    # Training loop
    # ----------------------------
    print('Starting training...')
    best = 1e10

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'

        start_time = time.time()

        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write(f'epoch: {epoch} |')

        for k, v in log_dict_train.items():
            logger.scalar_summary(f'train_{k}', v, epoch)
            logger.write(f'{k} {v:8f} | ')

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(
                os.path.join(opt.save_dir, f'model_{mark}.pth'),
                epoch,
                model,
                optimizer
            )

            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)

            for k, v in log_dict_val.items():
                logger.scalar_summary(f'val_{k}', v, epoch)
                logger.write(f'{k} {v:8f} | ')

            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(
                    os.path.join(opt.save_dir, 'model_best.pth'),
                    epoch,
                    model
                )
        else:
            save_model(
                os.path.join(opt.save_dir, 'model_last.pth'),
                epoch,
                model,
                optimizer
            )

        logger.write('\n')

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'Epoch {epoch} completed in {epoch_time:.2f} seconds')

        # Learning rate schedule
        if epoch in opt.lr_step:
            save_model(
                os.path.join(opt.save_dir, f'model_{epoch}.pth'),
                epoch,
                model,
                optimizer
            )
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop learning rate to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    logger.close()


# ----------------------------
# Bounding box rotation correction
# ----------------------------
def correct_rotate(preds2, height, width, rotate_degree):
    for pred in preds2:
        bboxs = preds2[pred][1]
        for bi in range(len(bboxs)):
            if rotate_degree == 90:
                x1_new = bboxs[bi][1]
                y1_new = width - bboxs[bi][2]
                x2_new = bboxs[bi][3]
                y2_new = width - bboxs[bi][0]
                score = bboxs[bi][4]
                bboxs[bi] = [x1_new, y1_new, x2_new, y2_new, score]
        preds2[pred][1] = bboxs
    return preds2


# ----------------------------
# Matching rate calculation using IoU
# ----------------------------
def caculate_matching_rate(preds, preds2, thres):
    all_box = 0
    match_box = 0

    for pred in preds:
        pred_bboxs = preds[pred][1]
        pred2_bboxs = preds2[pred][1]

        for bi in range(len(pred_bboxs)):
            if pred_bboxs[bi][4] >= thres:
                all_box += 1
            else:
                continue

            matched = False
            for bj in range(len(pred2_bboxs)):
                if pred2_bboxs[bj][4] < thres or matched:
                    continue

                overlap = IOU(pred2_bboxs[bj], pred_bboxs[bi])
                if overlap > 0.5:
                    match_box += 1
                    matched = True

    return all_box, match_box


# ----------------------------
# IoU computation for axis-aligned bounding boxes
# ----------------------------
def IOU(box1, gts):
    ixmin = np.maximum(gts[0], box1[0])
    iymin = np.maximum(gts[1], box1[1])
    ixmax = np.minimum(gts[2], box1[2])
    iymax = np.minimum(gts[3], box1[3])

    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)

    inters = iw * ih

    uni = (
        (box1[2] - box1[0] + 1.0) * (box1[3] - box1[1] + 1.0) +
        (gts[2] - gts[0] + 1.0) * (gts[3] - gts[1] + 1.0) -
        inters
    )

    overlaps = inters / uni
    return overlaps


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
