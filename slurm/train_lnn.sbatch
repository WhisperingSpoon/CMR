#!/bin/bash
#SBATCH --job-name=lnn-train
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# Trains the Lagrangian NN. Override hyper-parameters with env vars, e.g.
#   EPOCHS=300 WIDTH=256 sbatch slurm/train_lnn.sbatch
# Data-efficiency study (report): MAX_TRAIN_TRAJ=10 OUT=lnn_10traj sbatch slurm/train_lnn.sbatch
source slurm/env.sh
python experiments/train_lnn.py --data "$CMR_DATA/lnn" \
    --out "$CMR_RESULTS/${OUT:-lnn}" \
    --epochs "${EPOCHS:-200}" --width "${WIDTH:-128}" --depth "${DEPTH:-2}" \
    --batch "${BATCH:-1024}" --lr "${LR:-1e-3}" --stride "${STRIDE:-5}" \
    --max-train-traj "${MAX_TRAIN_TRAJ:-0}" --seed "${SEED:-0}" --device cuda
