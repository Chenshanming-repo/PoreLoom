#!/bin/bash
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH --partition=<gpu partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
# Incremental vs from-scratch timing on one V100 (exclusive GPU). SMOKE=1: tiny grid.
set -euo pipefail
W=/path/to/poreloom_work
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate poreloom-env
export PYTHONPATH=$W/PoreLoom/src OMP_NUM_THREADS=4
cd $W/PoreLoom
OUT=$W/runs/stream_timing/${TAG:-main}; mkdir -p $OUT
echo "$(date '+%F %T') host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
python experiments/20261016_stream_timing/timing.py run --data $W/data/r10_run01_p27000/ \
  --model $W/runs/budget_up/r10_run01/b18000_r0/stream/best.pt --out $OUT --device cuda:0 ${SMOKE:+--smoke}
echo "$(date '+%F %T') done"
