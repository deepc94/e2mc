#!/bin/bash
#SBATCH --job-name=simsiam_ft
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...) Always set to 1!
#SBATCH --gres=gpu:2          # This needs to match Trainer(devices=...) default 8
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...) default 8
#SBATCH --cpus-per-task=8     # This needs to match Trainer(num_workers=...) default 6 --> 3
#SBATCH --mem=60000          # 120000 for rtx-8000
#SBATCH --partition=gypsum-rtx8000
#SBATCH --time=3-00:00:00
#SBATCH --output=cluster_logs/output_%A.out

source ~/.bashrc
source activate e2mc
# debugging flags (optional)
# export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=12
# export WANDB_MODE="disabled"

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:9:4}
port=$(shuf -i49152-65535 -n1)
dist_url="tcp://${master_node}:${port}"

EXP_NAME=$1
FULL_EXP_NAME="${SLURM_JOB_ID}_${EXP_NAME}"
# EXPERIMENT_PATH="experiments/swav_400ep_bs256_continued_pretrain"
EXPERIMENT_PATH="experiments/${FULL_EXP_NAME}"
mkdir -p $EXPERIMENT_PATH
PRETRAINED_CKPT="experiments/simsiam_100ep_bs256_fullckpt.pth"
DATASET_PATH="data/ILSVRC2012"

srun --label python -u main_simsiam_ent_ft.py \
--exp-name $EXP_NAME \
--exp-dir $EXPERIMENT_PATH \
--data-path $DATASET_PATH \
--resume $PRETRAINED_CKPT \
--arch resnet50 \
--workers 8 \
--dim 2048 \
--pred-dim 512 \
--epochs $6 \
--batch-size 512 \
--lr $4 \
--pred-lr 0.05 \
--wd 0.0001 \
--dist-url $dist_url \
--fix-pred-lr \
--ent-coeff $2 \
--hypercov-coeff $3 \
--use-fp16 false \
--seed $5 \
2>&1
# --rank 0 \
# --world-size 1 \
# --use_fp16 true \

sleep 1
