#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import wandb

import simsiam.loader
import simsiam.builder
from src.utils import (
    bool_flag,
    fix_random_seeds,
    init_distributed_mode,
)

EPS = torch.finfo(torch.float32).eps

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--exp-name", type=str, default="test",
                    help='Name of the experiment')
parser.add_argument("--exp-dir", type=str, default="./exp",
                    help='Path to the experiment folder, where all logs/checkpoints will be stored')
parser.add_argument("--data-path", metavar='DIR', default='data/ILSVRC2012',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument("--use-fp16", type=bool_flag, default=False,
                    help="whether to train with mixed precision or not")
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--pred-lr', default=0.05, type=float,
                    help='predictor (fixed) learning rate', dest='pred_lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--dist-url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world-size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
# parser.add_argument('--world-size', default=-1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=31, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')
parser.add_argument("--ent-coeff", type=float, default=1.0,
                    help='Marginal Entropy regularization loss coefficient')
parser.add_argument("--hypercov-coeff", type=float, default=25.0,
                    help='Covariance regularization for joint entropy loss coefficient')


def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    if args.rank == 0:
        wandb.init(project="info-theory-ssl", name=args.exp_name, config=args)

    args.gpu = args.gpu_to_work_on

    # suppress printing if not master
    if args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = simsiam.builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim)

    # Apply SyncBN
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    print(model) # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda()

    init_lr = args.lr * args.batch_size / 256
    if args.fix_pred_lr:
        pred_lr = args.pred_lr * args.batch_size / 256
        optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                        {'params': model.predictor.parameters(), 'fix_lr': True, 'lr': pred_lr}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, lr=init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu_to_work_on]
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data_path, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    # train_dataset.samples = train_dataset.samples[:1024] ### debug
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=per_device_batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    # initialize entropy estimator
    ce_estimator = EntropyEstimator(args.batch_size, args.dim)
    # init mixed precision
    if args.use_fp16:
        print("Using Automatic Mixed Precision")
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, ce_estimator, scaler)

        if args.rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict() if args.use_fp16 else None,
            }, is_best=False, filename=os.path.join(args.exp_dir, f"checkpoint_{epoch+1:04d}.pth"))

    if args.rank == 0:
        wandb.finish()


def train(train_loader, model, criterion, optimizer, epoch, args, ce_estimator, scaler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    ss_losses = AverageMeter('SS_loss', ':.4f')
    ce_losses = AverageMeter('CE_loss', ':.4f')
    learning_rates = ConstantMeter('LR')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, ss_losses, ce_losses, losses, learning_rates],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        iteration = epoch * len(train_loader) + i
        learning_rates.update(optimizer.param_groups[0]["lr"])

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.use_fp16):
        # compute output and loss
            p1, p2, z1, z2 = model(x1=images[0], x2=images[1])

            ss_loss = -(criterion(p1, z2.detach()).mean() + criterion(p2, z1.detach()).mean()) * 0.5
            ss_losses.update(ss_loss.item(), images[0].size(0))

            # compute CE loss
            z1 = torch.cat(FullGatherLayer.apply(z1), dim=0)
            z2 = torch.cat(FullGatherLayer.apply(z2), dim=0)
            emb_std = (
                ce_estimator.get_std(z1.detach()).mean() / 2
                + ce_estimator.get_std(z2.detach()).mean() / 2
            )
            emb_mean = (
                ce_estimator.get_mean(z1.detach()).mean() / 2
                + ce_estimator.get_mean(z2.detach()).mean() / 2
            )
            ent_z1, hypercov_z1_loss = ce_estimator.get_ent_and_cov_loss(z1)
            ent_z2, hypercov_z2_loss = ce_estimator.get_ent_and_cov_loss(z2)
            ent_loss = ent_z1.mean() / 2 + ent_z2.mean() / 2
            hypercov_loss = hypercov_z1_loss + hypercov_z2_loss

            ce_loss = args.hypercov_coeff * hypercov_loss - args.ent_coeff * ent_loss
            ce_losses.update(ce_loss.item(), z1.size(0))

            # compute final loss and gradient
            loss = ss_loss + ce_loss
            losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0 and i % args.print_freq == 0:
            progress.display(i)

            wandb.log(dict(
                epoch=epoch,
                step=iteration,
                emb_std=emb_std,
                emb_mean=emb_mean,
                # hypemb_std=hypemb_std,
                ent_loss=ent_loss.detach(),
                hypercov_loss=hypercov_loss.detach(),
                ce_loss=ce_loss.detach(),
                ss_loss=ss_loss.detach(),
                loss=losses.val,
                lr=optimizer.param_groups[0]["lr"],
                pred_lr=optimizer.param_groups[1]["lr"],
            ))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ConstantMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def __str__(self):
        return f"{self.name} {self.val:.8f}"


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            continue
            # param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


class EntropyEstimator(object):
    """
    Estimator of joint entropy by maximing marginal entropuies 
    estimated using m-spacings formula, and minimizing off diagonal
    covariance terms.
    """
    def __init__(self, batch_size, num_features, epsilon=EPS):
        self.batch_size = batch_size
        self.num_features = num_features
        self.epsilon = epsilon
        print(f"Initializing entropy estimator with {self.batch_size=}")

    def get_cov_loss(self, x):
        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow(2).sum().div(self.num_features)
        return cov_loss

    def get_ent_and_cov_loss(self, x):
        # x_hyper = torch.sigmoid(x)
        # apply the 0-mean, unit variance gaussian cdf to the embedding distribution
        x_hyper = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
        x_hyper = x_hyper - x_hyper.mean(dim=0)
        ent = self.m_spacings_estimator(x_hyper)
        hypercov_loss = self.get_cov_loss(x_hyper)
        return (ent, hypercov_loss)

    def m_spacings_estimator(self, x):
        '''Calculates the marginal entropies of a D dimensional random variable
        with N samples (batch size), and a window of size m = sqrt(N)
            inputs: x (N, D)
            outputs: (D, )
        '''
        # print("Entropy embeddings: ", x.min().item(), x.max().item())
        N = x.shape[0]
        m = round(math.sqrt(N))  # TODO: make m fixed based on batch_size, not N
        x, _ = torch.sort(x, dim=0)
        x = x[m:] - x[:N - m]
        x = x * (N + 1) / m
        marginal_ents = torch.log(x + self.epsilon).sum(dim=0) / (N - m)
        return marginal_ents

    @staticmethod
    def get_mean(x):
        return x.mean(dim=0)

    @staticmethod
    def get_std(x):
        return x.std(dim=0)

    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


if __name__ == '__main__':
    main()
