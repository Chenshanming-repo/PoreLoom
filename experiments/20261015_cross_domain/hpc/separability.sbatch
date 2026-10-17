#!/bin/bash
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH --partition=<gpu partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
# Score separability (experiments/20261015_cross_domain/separability.py).
set -euo pipefail
W=/path/to/poreloom_work
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate poreloom-env
export PYTHONPATH=$W/PoreLoom/src OMP_NUM_THREADS=8
cd $W/PoreLoom
echo "$(date '+%F %T') host $(hostname)"
python experiments/20261015_cross_domain/separability.py --main $W/runs/main --data $W/data --fwd $W/runs/cross_domain \
  --rev $W/runs/cross_domain_rev --out $W/runs/separability --device cuda:0
echo "$(date '+%F %T') done"
