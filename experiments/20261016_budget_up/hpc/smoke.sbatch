#!/bin/bash
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH --partition=<gpu partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
# Smoke: budget 18000 on a 2000-read subset, 1 epoch, 3 betas (long cache, cap, memory).
set -euo pipefail
W=/path/to/poreloom_work
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate poreloom-env
export PYTHONPATH=$W/PoreLoom/src OMP_NUM_THREADS=8
cd $W/PoreLoom
python experiments/20261016_budget_up/budget_up.py run --data $W/data/r10_run01_p27000 \
  --stagea $W/runs/main/r10_run01/poreloom_r0/stageA/stage_A.pt --budget 18000 --rep 0 \
  --out $W/runs/budget_up/smoke/b18000 --limit 2000 --smoke-epochs 1 --device cuda:0
