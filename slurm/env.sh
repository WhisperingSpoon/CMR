#!/bin/bash
# Common environment for all jobs. Edit once for your cluster, then every .sbatch sources it.
#
# Hábrók (RUG) example -- check the exact module name with `module avail Python`:
#   module load Python/3.11.5-GCCcore-13.2.0
# Then create the venv once on a login node with slurm/setup_venv.sh.

set -euo pipefail

: "${CMR_ROOT:=$HOME/cmr}"                 # repository checkout on the cluster
: "${CMR_VENV:=$HOME/venvs/cmr}"            # virtual environment created by setup_venv.sh
: "${CMR_RESULTS:=$CMR_ROOT/results}"       # outputs (scratch is a good place on most clusters)
: "${CMR_DATA:=$CMR_ROOT/data}"
export CMR_ROOT CMR_VENV CMR_RESULTS CMR_DATA
export CMR_CACHE="${CMR_CACHE:-$CMR_ROOT/.cache}"   # generated URDFs

module purge 2>/dev/null || true
# module load Python/3.11.5-GCCcore-13.2.0         # <-- uncomment / adapt
source "$CMR_VENV/bin/activate"
cd "$CMR_ROOT"

# one thread per process: parallelism comes from Slurm tasks / SubprocVecEnv
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$CMR_RESULTS" "$CMR_DATA" logs
