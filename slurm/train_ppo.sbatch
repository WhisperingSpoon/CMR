#!/bin/bash
#SBATCH --job-name=ppo-peg
#SBATCH --array=0-2
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Three PPO seeds (report mean +- spread), 16 parallel environments each, CPU only.
# ~300 env steps/s per core -> 3 M steps take roughly 15-30 min of wall time per 16-core task
# plus PPO updates; the 24 h limit is generous. Resubmitting resumes from the last checkpoint.
source slurm/env.sh
python experiments/train_ppo.py --seed "$SLURM_ARRAY_TASK_ID" --n-envs "$SLURM_CPUS_PER_TASK" \
    --timesteps "${TIMESTEPS:-3000000}" --max-offset "${MAX_OFFSET:-0.005}" \
    --out "$CMR_RESULTS/ppo/seed$SLURM_ARRAY_TASK_ID" --resume
