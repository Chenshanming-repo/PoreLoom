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
# One native_cp.py run (4 main-run baselines + C-p) on dataset DS, replicate REP. Re-submitting resumes.
set -euo pipefail
W=/path/to/poreloom_work
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate poreloom-env
export PYTHONPATH=$W/PoreLoom/src OMP_NUM_THREADS=8
cd $W/PoreLoom
FS=$(python -c "import json;print(json.load(open('$W/runs/main/$DS/poreloom_r$REP/run.json'))['fs'])")
echo "$(date '+%F %T') host $(hostname) ds $DS rep $REP fs $FS"
python experiments/20261015_window_tools/native_cp.py run --data $W/data/$DS --main $W/runs/main/$DS --fs $FS --rep $REP \
  --out $W/runs/native_cp/$DS/r$REP --device cuda:0
echo "$(date '+%F %T') done"
