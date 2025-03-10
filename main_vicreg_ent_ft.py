# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets

import augmentations as aug
from distributed import init_distributed_mode

import resnet
import wandb

EPS = torch.finfo(torch.float32).eps


def get_arguments():
    parser = argparse.ArgumentParser(description="Fintune an SSL model with entropy maximization", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=True,
                        help='Path to the image net dataset')

    # Checkpoints
    parser.add_argument("--exp-name", type=str, default="test",
                        help='Name of the experiment')
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')
    parser.add_argument("--resume-ckpt", type=Path, default=None,
                        help='Explicit path to model checkpoint for resuming training or finetuning')

    # Model
    parser.add_argument("--arch", type=str, default="resnet50",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="8192-8192-8192",
                        help='Size and number of layers of the MLP expander head')

    # Optim
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=2048,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')
    parser.add_argument("--ent-coeff", type=float, default=10.0,
                        help='Entropy regularization loss coefficient')
    parser.add_argument("--hypercov-coeff", type=float, default=1.0,
                        help='Covariance regularization for joint entropy loss coefficient')
    parser.add_argument("--epsilon", type=float, default=EPS,
                        help='epsilon value to use in loss computation')

    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument("--random-seed", type=int, default=None)

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    torch.backends.cudnn.benchmark = True
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        print(f"Setting Random seeds to {args.random_seed}")

    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)
        wandb.init(project="info-theory-ssl", name=args.exp_name, config=args)

    transforms = aug.TrainTransform()

    dataset = datasets.ImageFolder(args.data_dir / "train", transforms)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    model = ICEReg(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    elif args.resume_ckpt.is_file():
        if args.rank == 0:
            print(f"resuming from checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, ((x, y), _) in enumerate(loader, start=epoch * len(loader)):
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)

            lr = adjust_learning_rate(args, optimizer, loader, step, fixed=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_dict = model.forward(x, y)
            scaler.scale(loss_dict['loss']).backward()
            scaler.step(optimizer)
            scaler.update()

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                # print essential info to the output
                stats = dict(
                    epoch=epoch,
                    step=step,
                    emb_std=loss_dict['emb_std'].item(),
                    repr_loss=loss_dict['repr_loss'].item(),
                    std_loss=loss_dict['std_loss'].item(),
                    cov_loss=loss_dict['cov_loss'].item(),
                    ent_loss=loss_dict['ent_loss'].item(),
                    hypercov_loss=loss_dict['hypercov_loss'].item(),
                    vicreg_loss=loss_dict['vicreg_loss'].item(),
                    ce_loss=loss_dict['ce_loss'].item(),
                    loss=loss_dict['loss'].item(),
                    time=int(current_time - start_time),
                    lr=lr,
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)

                # log info to wandb
                wandb.log(stats)
                last_logging = current_time
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / f"model_epoch{epoch+1:04d}.pth")
            # torch.save(state, args.exp_dir / "model.pth")
    if args.rank == 0:
        # torch.save(model.module.backbone.state_dict(), args.exp_dir / "resnet50.pth")
        wandb.finish()


def adjust_learning_rate(args, optimizer, loader, step, fixed=False):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if not fixed:
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            step -= warmup_steps
            max_steps -= warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)
    else:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class ICEReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)

        #################### Entropy regularizer #####################
        ent_x, hypercov_x_loss = self._get_ent_and_cov_loss(x)
        ent_y, hypercov_y_loss = self._get_ent_and_cov_loss(y)
        ent_loss = ent_x.mean() / 2 + ent_y.mean() / 2
        hypercov_loss = hypercov_x_loss + hypercov_y_loss

        ce_loss = (
            self.args.hypercov_coeff * hypercov_loss
            - self.args.ent_coeff * ent_loss
        )
        ###############################################################
        #################### Original VICReg loss #####################
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = self._get_std(x)
        std_y = self._get_std(y)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        emb_std = std_x.detach().mean() / 2 + std_y.detach().mean() / 2  # only for plotting

        cov_loss = self._get_cov_loss(x) + self._get_cov_loss(y)

        vicreg_loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        ################################################################

        loss = vicreg_loss + ce_loss

        loss_dict = {
            'emb_std': emb_std,
            'repr_loss': repr_loss.detach(),
            'cov_loss': cov_loss.detach(),
            'std_loss': std_loss.detach(),
            'ent_loss': ent_loss.detach(),
            'hypercov_loss': hypercov_loss.detach(),
            'vicreg_loss': vicreg_loss.detach(),
            'ce_loss': ce_loss.detach(),
            'loss': loss,
        }

        return loss_dict

    def _get_cov_loss(self, x):
        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features)
        return cov_loss

    def _get_std(self, x):
        return torch.sqrt(x.var(dim=0) + 0.0001)

    def _get_ent_and_cov_loss(self, x):
        x_hyper = torch.sigmoid(x)
        x_hyper = x_hyper - x_hyper.mean(dim=0)
        ent = self.m_spacings_estimator(x_hyper)
        hypercov_loss = self._get_cov_loss(x_hyper)
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
        marginal_ents = torch.log(x + self.args.epsilon).sum(dim=0) / (N - m)
        return marginal_ents


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


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


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('ICEReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
