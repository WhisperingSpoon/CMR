#!/bin/bash
# Submit all training with dependencies:   bash slurm/submit_training.sh
#   1. LNN data collection (array)  ->  2. LNN training (GPU)
#   3. PPO training (array of seeds, independent of 1-2)
set -euo pipefail
mkdir -p logs
DATA=$(sbatch --parsable slurm/collect_lnn.sbatch)
LNN=$(sbatch --parsable --dependency=afterok:$DATA slurm/train_lnn.sbatch)
PPO=$(sbatch --parsable slurm/train_ppo.sbatch)
echo "data collection: $DATA   LNN training: $LNN (after $DATA)   PPO: $PPO"
echo "When finished, copy back:  results/lnn/lnn_best.pt  and  results/ppo/seed*/best_model.zip"
