#!/bin/bash
#SBATCH --job-name=lnn-data
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
# 10 tasks x 7 trajectories x 20 s = 70 trajectories (Liu et al. 2024 used 70 for the Panda).
# Each 20 s trajectory takes roughly 20-40 s of compute.
source slurm/env.sh
python experiments/collect_lnn_data.py --n-traj 7 --duration 20 --seed $((1000 + SLURM_ARRAY_TASK_ID)) \
    --out "$CMR_DATA/lnn"
