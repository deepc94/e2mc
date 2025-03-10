#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...) Always set to 1!
#SBATCH --gres=gpu:4          # This needs to match Trainer(devices=...) default 8
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...) default 8
#SBATCH --cpus-per-task=6     # This needs to match Trainer(num_workers=...) default 6 --> 3
#SBATCH --mem=180000          # 120000 for rtx-8000
#SBATCH --partition=gypsum-1080ti
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
dist_url="tcp://"
dist_url+=$master_node
dist_url+=":$(shuf -i29400-40000 -n1)"

EXP_NAME=$1
FULL_EXP_NAME="${SLURM_JOB_ID}_${EXP_NAME}"
EXPERIMENT_PATH="experiments/${FULL_EXP_NAME}"
mkdir -p $EXPERIMENT_PATH
DATASET_PATH="data/ILSVRC2012"

srun --label python -u eval_linear.py \
--exp_name $EXP_NAME \
--data_path $DATASET_PATH \
--train_percent $2 \
--dump_path $EXPERIMENT_PATH \
--pretrained $6 \
--workers 6 \
--epochs 100 \
--batch_size 64 \
--lr $3 \
--wd $4 \
--dist_url $dist_url \
--arch resnet50 \
--tags $7 \
--seed $5 \
--scheduler_type cosine \
2>&1

sleep 1