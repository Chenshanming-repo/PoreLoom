#!/bin/bash
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH --partition=<gpu partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# One 6000-window baseline: METHOD (readcurrent | nanodeep_upstream | squigglenet) on DS, replicate REP. Re-submitting resumes.
set -euo pipefail
W=/path/to/poreloom_work
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate poreloom-env
export PYTHONPATH=$W/PoreLoom/src OMP_NUM_THREADS=8
cd $W/PoreLoom
FS=$(python -c "import json;print(json.load(open('$W/runs/main/$DS/poreloom_r$REP/run.json'))['fs'])")
RECIPE=project; [ "$METHOD" = nanodeep_upstream ] && RECIPE=upstream
OUT=$W/runs/win6000/$DS/${METHOD}_r$REP
echo "$(date '+%F %T') host $(hostname) method $METHOD ds $DS rep $REP fs $FS"
python experiments/20261016_window6000/win6000.py run --tool ${METHOD%%_*} --recipe $RECIPE --data $W/data/$DS --fs $FS \
  --rep $REP --out $OUT --device cuda:0
touch $OUT/JOB_DONE
echo "$(date '+%F %T') done"
