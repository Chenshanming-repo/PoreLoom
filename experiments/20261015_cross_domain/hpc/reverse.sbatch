#!/bin/bash
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH --partition=<gpu partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# Reverse cross-dataset test (experiments/20261015_cross_domain/reverse.py); resumable.
set -euo pipefail
W=/path/to/poreloom_work
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate poreloom-env
export PYTHONPATH=$W/PoreLoom/src OMP_NUM_THREADS=8
cd $W/PoreLoom
echo "$(date '+%F %T') host $(hostname)"
X=experiments/20261015_cross_domain/reverse.py
python $X run --main $W/runs/main --data $W/data --out $W/runs/cross_domain_rev --device cuda:0
python $X report --main $W/runs/main --out $W/runs/cross_domain_rev --splits $W/data/sputum_splits --md $W/runs/cross_domain_rev/summary_zh.md
echo "$(date '+%F %T') done"
