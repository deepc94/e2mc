# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import math
import os
from pathlib import Path
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import apex
from apex.parallel.LARC import LARC

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
from src.multicropdataset import MultiCropDataset
import src.resnet50 as resnet_models
import wandb

EPS = torch.finfo(torch.float32).eps

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of SwAV")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="data/ILSVRC2012/train",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=3000, type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")
parser.add_argument("--mmcr_weight", type=float, default=1.0,
                    help='weight to use for the uniformity loss')
# parser.add_argument("--kernel_width", type=float, default=3.0,
#                     help='the value of t in the gaussian potential')
# parser.add_argument("--ent_coeff", type=float, default=10.0,
#                     help='Marginal Entropy regularization loss coefficient')
# parser.add_argument("--hypercov_coeff", type=float, default=1.0,
#                     help='Covariance regularization for joint entropy loss coefficient')

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=8, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=1,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--exp_name", type=str, default="test",
                    help='Name of the experiment')
parser.add_argument("--resume_ckpt", type=Path, default=None,
                    help='Explicit path to model checkpoint for resuming training or finetuning')
parser.add_argument("--seed", type=int, default=31, help="seed")


def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")
    if args.rank == 0:
        wandb.init(project="info-theory-ssl", name=args.exp_name, config=args)

    # build data
    train_dataset = MultiCropDataset(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        # size_dataset=1024,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        return_unnorm_emb=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes,
    )
    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        logger.info("Initializing mixed precision done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on]
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    if os.path.isfile(os.path.join(args.dump_path, "checkpoint.pth.tar")):
        restart_from_checkpoint(
            os.path.join(args.dump_path, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=model,
            optimizer=optimizer,
            amp=apex.amp,
        )
    elif args.resume_ckpt.is_file():
        restart_from_checkpoint(
            args.resume_ckpt,
            run_variables=to_restore,
            state_dict=model,
        )
    start_epoch = to_restore["epoch"]

    # build the queue
    queue = None
    queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

    # initialize entropy estimator
    ce_estimator = EntropyEstimator(args.batch_size * args.world_size, args.feat_dim)

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
                args.feat_dim,
            ).cuda()

        # train the network
        scores, queue = train(train_loader, model, optimizer, epoch, lr_schedule, queue, ce_estimator)
        training_stats.update(scores)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_fp16:
                save_dict["amp"] = apex.amp.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    # os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                    os.path.join(args.dump_checkpoints, f"model_epoch{epoch+1:04d}.pth"),
                )
        if queue is not None:
            torch.save({"queue": queue}, queue_path)

    if args.rank == 0:
        wandb.finish()


def train(train_loader, model, optimizer, epoch, lr_schedule, queue, ce_estimator):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    swav_losses = AverageMeter()
    mmcr_losses = AverageMeter()

    model.train()
    use_the_queue = False

    end = time.time()
    for it, inputs in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with torch.no_grad():
            w = model.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)

        # ============ multi-res forward passes ... ============
        z, embedding, output = model(inputs)  # inputs is a list of different views
        z = z.detach()
        # embedding = embedding.detach()
        bs = inputs[0].size(0)

        # ============ swav loss ... ============
        swav_loss = 0
        hypemb_std = 0
        hypemb_mean = 0
        for i, crop_id in enumerate(args.crops_for_assign):
            # crop_for_assign decides which out of the [0, 1, 2, ..., 6, 7] views should
            # be used for computing the codes Q. In this case, only the 0 or 1 view is used.
            with torch.no_grad():
                hypemb_std += (ce_estimator.get_std(
                        embedding[crop_id * bs: (crop_id + 1) * bs].detach()
                    ).mean() / len(args.crops_for_assign))
                hypemb_mean += (ce_estimator.get_mean(
                        embedding[crop_id * bs: (crop_id + 1) * bs].detach()
                    ).mean() / len(args.crops_for_assign))

                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat((torch.mm(
                            queue[i],
                            model.module.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs].detach()

                # get assignments
                q = distributed_sinkhorn(out)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / args.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            swav_loss += subloss / (np.sum(args.nmb_crops) - 1)
        swav_loss /= len(args.crops_for_assign)

        # ============ MMCR loss ... ============
        # embedding is (B*n, 128) dimensional
        total_views = np.sum(args.nmb_crops)  # 8
        # print(f"Shape before stacking: {embedding.shape=}")
        # reshape into (B, 128, n) where the n views are separated to a new dimension
        embedding = torch.stack([
            embedding[bs * view_id: bs * (view_id + 1)] for view_id in range(total_views)
        ], dim=-1)  # (B, 128, n)
        # print(f"Shape after stacking: {embedding.shape=}")
        # aggregate the batch of embeddings from other GPUs
        embedding = torch.cat(FullGatherLayer.apply(embedding), dim=0)  # (2*B, 128, n)
        # print(f"Shape after gathering: {embedding.shape=}")
        # take average across the views to compute the centroids
        centroids = embedding.mean(dim=-1)  # (2*B, 128)
        # print(f"Shape after taking mean: {centroids.shape=}")
        # filter infs and nans
        selected = centroids.isfinite().all(dim=1)
        centroids = centroids[selected]
        if selected.sum() != centroids.shape[0]:
            print("filtered nan")
        mmcr_loss = -torch.linalg.svdvals(centroids).sum()

        # ============ final loss ... ============
        loss = swav_loss + args.mmcr_weight * mmcr_loss

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # cancel gradients for the prototypes
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        optimizer.step()

        # ============ misc ... ============
        swav_losses.update(swav_loss.item(), inputs[0].size(0))
        mmcr_losses.update(mmcr_loss.item(), inputs[0].size(0))
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank == 0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "hypemb_std {hypemb_std}\t"
                "Swav Loss {swav_loss.val:.4f} ({swav_loss.avg:.4f})\t"
                "MMCR Loss {mmcr_loss.val:.4f} ({mmcr_loss.avg:.4f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    hypemb_std=hypemb_std.item(),
                    swav_loss=swav_losses,
                    mmcr_loss=mmcr_losses,
                    loss=losses,
                    lr=optimizer.optim.param_groups[0]["lr"],
                )
            )
            wandb.log(dict(
                epoch=epoch,
                step=iteration,
                hypemb_std=hypemb_std,
                hypemb_mean=hypemb_mean,
                # emb_std=emb_std,
                # emb_mean=emb_mean,
                swav_loss=swav_loss.detach(),
                mmcr_loss=mmcr_loss.detach(),
                loss=losses.val,
                lr=optimizer.optim.param_groups[0]["lr"],
            ))
    return (epoch, losses.avg), queue


@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


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


if __name__ == "__main__":
    main()
