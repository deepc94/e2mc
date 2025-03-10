#!/bin/bash
#SBATCH --job-name=swav_vc_ft
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...) Always set to 1!
#SBATCH --gres=gpu:2          # This needs to match Trainer(devices=...) default 8
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...) default 8
#SBATCH --cpus-per-task=8     # This needs to match Trainer(num_workers=...) default 6 --> 3
#SBATCH --mem=60000          # 120000 for rtx-8000
#SBATCH --partition=gypsum-rtx8000
#SBATCH --time=5-00:00:00
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
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

EXP_NAME=$1
FULL_EXP_NAME="${SLURM_JOB_ID}_${EXP_NAME}"
# EXPERIMENT_PATH="experiments/swav_400ep_bs256_continued_pretrain"
EXPERIMENT_PATH="experiments/${FULL_EXP_NAME}"
mkdir -p $EXPERIMENT_PATH
# PRETRAINED_CKPT="experiments/swav_400ep_bs256_fullckpt.pth"
PRETRAINED_CKPT="experiments/swav_800ep_bs4096_fullckpt.pth"
DATASET_PATH="data/ILSVRC2012/train"

srun --label python -u main_swav_vc_ft.py \
--exp_name $EXP_NAME \
--data_path $DATASET_PATH \
--resume_ckpt $PRETRAINED_CKPT \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 3000 \
--queue_length 3584 \
--epoch_queue_starts 7 \
--epochs $7 \
--batch_size 256 \
--base_lr $5 \
--final_lr 0.0006 \
--freeze_prototypes_niters 2503 \
--wd 0.000001 \
--warmup_epochs 0 \
--dist_url $dist_url \
--arch resnet50 \
--use_fp16 true \
--sync_bn pytorch \
--dump_path $EXPERIMENT_PATH \
--std_coeff $3 \
--cov_coeff $4 \
--reg_space $2 \
--seed $6 \
2>&1

sleep 1
