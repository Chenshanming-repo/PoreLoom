#!/bin/bash
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH --partition=<gpu partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# One ablation job (experiments/20261015_ablation/ablation.py): STAGE steps | fixedwin on dataset DS, replicate REP
# (fixedwin: fixed_window.py run). Re-submitting resumes: finished parts are skipped.
set -euo pipefail
W=/path/to/poreloom_work
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate poreloom-env
export PYTHONPATH=$W/PoreLoom/src OMP_NUM_THREADS=8
cd $W/PoreLoom
FS=$(python -c "import json;print(json.load(open('$W/runs/main/$DS/poreloom_r$REP/run.json'))['fs'])")
echo "$(date '+%F %T') host $(hostname) stage $STAGE ds $DS rep $REP fs $FS"
A=experiments/20261015_ablation/ablation.py
case $STAGE in
  steps)   python $A steps --main $W/runs/main/$DS --data $W/data/$DS --rep $REP --out $W/runs/ablation/$DS/r$REP --device cuda:0 ;;
  fixedwin) python experiments/20261015_ablation/fixed_window.py run --main $W/runs/main/$DS --data $W/data/$DS --rep $REP --out $W/runs/fixed_window/$DS/r$REP --device cuda:0 ;;
esac
echo "$(date '+%F %T') done"
