# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import time
from logging import getLogger
import urllib
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from apex.parallel.LARC import LARC
from custom_datasets import INaturalist2018

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    accuracy,
    exclude_bias_and_norm,
)
import src.resnet50 as resnet_models
import wandb


logger = getLogger()


def list_of_strings(arg):
    return arg.split(',')

parser = argparse.ArgumentParser(description="Evaluate models: Linear classification on ImageNet")

#########################
#### main parameters ####
#########################
parser.add_argument("--exp_name", type=str, default="probe",
                    help='Name of the experiment')
parser.add_argument("--tags", type=list_of_strings, required=False, default=None,
                    help="(optional) Pass experiment identifiers as comma separated keywords")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--dataset", type=str, default="ImageNet", choices=('ImageNet', 'INaturalist2018'),
                    help="Name of dataset")
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--train_percent", type=int, default=100, choices=(100, 50, 10, 1),
                    help="size of training set in percent")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")

#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained weights")
parser.add_argument("--global_pooling", default=True, type=bool_flag,
                    help="if True, we use the resnet50 global average pooling")
parser.add_argument("--use_bn", default=False, type=bool_flag,
                    help="optionally add a batchnorm layer before the linear classifier")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--lr", default=0.3, type=float, help="initial learning rate")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--nesterov", default=False, type=bool_flag, help="nesterov momentum")
parser.add_argument("--lars", default=False, type=bool_flag, help="whether to use LARS optimizer")
parser.add_argument("--scheduler_type", default="cosine", type=str, choices=["step", "cosine"])
# for multi-step learning rate decay
parser.add_argument("--decay_epochs", type=int, nargs="+", default=[60, 80],
                    help="Epochs at which to decay learning rate.")
parser.add_argument("--gamma", type=float, default=0.1, help="decay factor")
# for cosine learning rate schedule
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")


def main():
    global args, best_acc1, best_acc5
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(
        args, "epoch", "loss", "prec1", "prec5", "loss_val", "prec1_val", "prec5_val"
    )
    if args.rank == 0:
        wandb.init(project="info-theory-ssl", name=args.exp_name, config=args, tags=args.tags)

    # build data
    if args.dataset == "INaturalist2018":
        train_dataset = INaturalist2018(args.data_path, train=True)
        if args.train_percent in {10, 50}:
            train_dataset.index = []
            with open(args.data_path / f"{args.train_percent}percent.txt", 'r') as f:
                train_files = f.readlines()
            for fname in train_files:
                fname = fname.strip()
                cat_id = int(fname.split(os.sep)[-2])
                train_dataset.index.append((fname, cat_id))
        val_dataset = INaturalist2018(args.data_path, train=False)
    else:
        train_data_path = os.path.join(args.data_path, "train")
        train_dataset = datasets.ImageFolder(train_data_path)
        if args.train_percent in {1, 10}:
            # take either 1% or 10% of images
            subset_file = urllib.request.urlopen(f"https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.train_percent}percent.txt")
            list_imgs = [li.decode("utf-8").split('\n')[0] for li in subset_file]
            train_dataset.samples = [(
                os.path.join(train_data_path, li.split('_')[0], li),
                train_dataset.class_to_idx[li.split('_')[0]]
            ) for li in list_imgs]
        val_dataset = datasets.ImageFolder(os.path.join(args.data_path, "val"))

    tr_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )
    train_dataset.transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        tr_normalize,
    ])
    val_dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        tr_normalize,
    ])
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    logger.info(f"Building {args.dataset} data done with {len(train_dataset)} training images and {len(val_dataset)} validation images loaded.")

    # build model
    model = resnet_models.__dict__[args.arch](output_dim=0, eval_mode=True)
    num_classes = 1000 if args.dataset == "ImageNet" else 8142
    linear_classifier = RegLog(num_classes, args.arch, args.global_pooling, args.use_bn)
    # convert batch norm layers (if any)
    linear_classifier = nn.SyncBatchNorm.convert_sync_batchnorm(linear_classifier)
    logger.info(f"Building linear classifier with {linear_classifier.linear.in_features=} and {linear_classifier.linear.out_features=} done.")

    # model to gpu
    model = model.cuda()
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(
        linear_classifier,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )
    model.eval()

    # load weights
    if os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location="cuda:" + str(args.gpu_to_work_on))
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict_remap = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("module.backbone."):
                state_dict_remap[key.replace("module.backbone.", "")] = value
            elif key.startswith("backbone."):
                state_dict_remap[key.replace("backbone.", "")] = value
            elif key.startswith("module.encoder_q."):
                state_dict_remap[key.replace("module.encoder_q.", "")] = value
            elif key.startswith("module.encoder.") and not key.startswith("module.encoder.fc"):
                state_dict_remap[key.replace("module.encoder.", "")] = value
            elif key.startswith("module."):
                state_dict_remap[key.replace("module.", "")] = value
            else:
                state_dict_remap[key] = value
        state_dict = state_dict_remap

        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                logger.info('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info("Load pretrained model with msg: {}".format(msg))
    else:
        logger.info("No pretrained weights found => training with random weights")

    # scale learning rate
    args.lr = args.lr * (args.batch_size * args.world_size / 256)

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        lr=args.lr,
        nesterov=args.nesterov,
        momentum=0.9,
        weight_decay=args.wd,
    )
    if args.lars:
        logger.info("Using LARS optimizer and Cosine Scheduler...")
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
        scheduler = Scheduler(args.lr, args.epochs)
    elif args.scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, args.decay_epochs, gamma=args.gamma
        )
    elif args.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=args.final_lr
        )

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc1": 0., "best_acc5": 0.}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc1 = to_restore["best_acc1"]
    best_acc5 = to_restore["best_acc5"]
    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set samplers
        train_loader.sampler.set_epoch(epoch)

        # take scheduler step for lars before calling train
        if args.lars:
            scheduler.step(optimizer, epoch)

        scores = train(model, linear_classifier, optimizer, train_loader, epoch)
        scores_val = validate_network(val_loader, model, linear_classifier, epoch)
        training_stats.update(scores + scores_val)

        if not args.lars:
            scheduler.step()

        # save checkpoint
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                # "scheduler": scheduler.state_dict(),
                "best_acc1": best_acc1,
                "best_acc5": best_acc5,
            }
            if not args.lars:
                save_dict.update({"scheduler": scheduler.state_dict()})
            torch.save(save_dict, os.path.join(args.dump_path, "checkpoint.pth.tar"))
    logger.info("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc1))
    if args.rank == 0:
        wandb.finish()


class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, num_labels, arch="resnet50", global_avg=False, use_bn=True):
        super(RegLog, self).__init__()
        self.bn = None
        if global_avg:
            if arch == "resnet50":
                s = 2048
            elif arch == "resnet50w2":
                s = 4096
            elif arch == "resnet50w4":
                s = 8192
            self.av_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            assert arch == "resnet50"
            s = 8192
            self.av_pool = nn.AvgPool2d(6, stride=1)
            if use_bn:
                self.bn = nn.BatchNorm2d(2048)
        self.linear = nn.Linear(s, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # average pool the final feature map
        x = self.av_pool(x)

        # optional BN
        if self.bn is not None:
            x = self.bn(x)

        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def train(model, reglog, optimizer, loader, epoch):
    """
    Train the models on the dataset.
    """
    # running statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    end = time.perf_counter()

    model.eval()
    reglog.train()
    criterion = nn.CrossEntropyLoss().cuda()

    for iter_epoch, (inp, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model(inp)
        output = reglog(output)

        # compute cross entropy loss
        loss = criterion(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # update stats
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inp.size(0))
        top1.update(acc1[0], inp.size(0))
        top5.update(acc5[0], inp.size(0))

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        # verbose
        if args.rank == 0 and iter_epoch % 50 == 0:
            logger.info(
                "Epoch[{0}] - Iter: [{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec {top1.val:.3f} ({top1.avg:.3f})\t"
                "LR {lr}".format(
                    epoch,
                    iter_epoch,
                    len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
            stats = dict(
                epoch=epoch,
                step=(epoch * len(loader) + iter_epoch),
                lr_head=optimizer.param_groups[0]["lr"],
                loss=losses.val,
            )
            wandb.log(stats)

    return epoch, losses.avg, top1.avg.item(), top5.avg.item()


def validate_network(val_loader, model, linear_classifier, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global best_acc1
    global best_acc5

    # switch to evaluate mode
    model.eval()
    linear_classifier.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        end = time.perf_counter()
        for i, (inp, target) in enumerate(val_loader):

            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = linear_classifier(model(inp))
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1[0], inp.size(0))
            top5.update(acc5[0], inp.size(0))

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

    if top1.avg.item() > best_acc1:
        best_acc1 = top1.avg.item()
    if top5.avg.item() > best_acc5:
        best_acc5 = top5.avg.item()

    if args.rank == 0:
        logger.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "Acc@1 {top1.avg:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                batch_time=batch_time, loss=losses, top1=top1, acc=best_acc1))
        stats = dict(
            epoch=epoch,
            val_loss=losses.avg,
            acc1=top1.avg,
            acc5=top5.avg,
            best_acc1=best_acc1,
            best_acc5=best_acc5,
        )
        wandb.log(stats)

    return losses.avg, top1.avg.item(), top5.avg.item()


class Scheduler(object):
    def __init__(self, init_lr, max_epochs):
        self.max_epochs = max_epochs
        self.init_lr = init_lr

    def step(self, optimizer, epoch):
        """Decay the learning rate based on schedule"""
        cur_lr = self.init_lr * 0.5 * (1. + math.cos(math.pi * epoch / self.max_epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr


if __name__ == "__main__":
    main()
