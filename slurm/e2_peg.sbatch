#!/bin/bash
#SBATCH --job-name=e2-peg
#SBATCH --array=0-7
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Optional: run the E2 evaluation on the cluster, split into 8 shards (trials_shard*.csv).
source slurm/env.sh
python experiments/e2_peg_in_hole.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards 8 \
    --trials "${TRIALS:-30}" --ppo-model "$CMR_RESULTS/ppo/seed${PPO_SEED:-0}/best_model.zip" \
    --out "$CMR_RESULTS/e2"
