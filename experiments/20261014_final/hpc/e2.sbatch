#!/bin/bash
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH --partition=<gpu partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
# E2 (cascade.py run) for one dataset DS and replicate REP: adaptive cascade of every finished baseline + PoreLoom TEST curve.
set -euo pipefail
W=/path/to/poreloom_work
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate poreloom-env
export PYTHONPATH=$W/PoreLoom/src OMP_NUM_THREADS=8
cd $W/PoreLoom
echo "$(date '+%F %T') host $(hostname) ds $DS rep $REP"
python experiments/20261014_final/cascade.py run --main $W/runs/main/$DS --data $W/data/$DS --out $W/runs/e2/$DS --rep $REP --device cuda:0
echo "$(date '+%F %T') done"
