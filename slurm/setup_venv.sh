#!/bin/bash
# One-time setup on a login node:   bash slurm/setup_venv.sh
# pybullet has no wheel for Python >= 3.13 and compiles from source (~15 min); Python 3.10-3.12
# installs a prebuilt wheel. For GPU training of the LNN, install the CUDA build of torch that
# matches the cluster driver (default PyPI wheel is usually fine).
set -euo pipefail
: "${CMR_VENV:=$HOME/venvs/cmr}"
# module load Python/3.11.5-GCCcore-13.2.0         # <-- same module as in env.sh
python3 -m venv "$CMR_VENV"
source "$CMR_VENV/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -c "import pybullet, torch, stable_baselines3; print('ok', torch.__version__, torch.cuda.is_available())"
