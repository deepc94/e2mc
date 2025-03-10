#!/bin/bash -l
#SBATCH --job-name=ICERegFT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...) Always set to 1!
#SBATCH --gres=gpu:2          # This needs to match Trainer(devices=...) default 8
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...) default 8
#SBATCH --cpus-per-task=8     # This needs to match Trainer(num_workers=...) default 6 --> 3
#SBATCH --mem=60000          # 120000 for rtx-8000
#SBATCH --partition=gpu
#SBATCH --constraint=l40s
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster_logs/output_%A.out

source ~/.bashrc
source activate e2mc
# debugging flags (optional)
# export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=12
master_addr=$(hostname -s)
master_port=$(shuf -i29400-40000 -n1)
echo "Using master addr: ${master_addr} and master port: ${master_port}"
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

exp_name=$1
full_exp_name="${SLURM_JOB_ID}_${exp_name}"

srun python3 -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --node_rank=0 --master_addr=${master_addr} --master_port=${master_port} main_vicreg_ent_ft.py \
    --exp-name ${exp_name} \
    --data-dir data/ILSVRC2012 \
    --exp-dir experiments/${full_exp_name} \
    --arch resnet50 \
    --epochs $6 \
    --batch-size 512 \
    --world-size 2 \
    --base-lr $4 \
    --sim-coeff 25 \
    --std-coeff 25 \
    --cov-coeff 1 \
    --ent-coeff $2 \
    --hypercov-coeff $3 \
    --num-workers 8 \
    --resume-ckpt experiments/resnet50_fullckpt.pth \
    --random-seed $5 \
    --epsilon 1e-7 \
2>&1

sleep 1
