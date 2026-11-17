from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
import time

def main(opt):

    torch.manual_seed(opt.seed)

    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    logger = Logger(opt)

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

    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data loaders...')

    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

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

    if opt.test:
        start_time = time.time()

        with torch.no_grad():
            _, preds = trainer.val(0, test_loader)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Executed in: {execution_time:.2f} seconds")

        test_loader.dataset.run_keypoint_eval(preds, opt.save_dir)
        return

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

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

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
