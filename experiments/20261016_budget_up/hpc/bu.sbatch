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
# Dataset DS (sampling rate FS, long cache data/${DS}_p$P).
# STAGE=stagea: method stage A of replicate REP on the default 9000 cache; STAGE=run: budget BUDGET, replicate REP;
# REF=1: the main run's S1 + head copied into the run directory (only TEST is replayed). Re-submitting resumes.
set -euo pipefail
W=/path/to/poreloom_work
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate poreloom-env
export PYTHONPATH=$W/PoreLoom/src OMP_NUM_THREADS=8
cd $W/PoreLoom
R=$W/runs/budget_up/$DS; MAIN=$W/runs/main/$DS/poreloom_r0
echo "$(date '+%F %T') host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) ds $DS stage $STAGE budget ${BUDGET:-} rep $REP ref ${REF:-0}"
if [ "$STAGE" = stagea ]; then
  python -m poreloom.method stageA --data $W/data/$DS --out $R/stageA_r$REP --fs $FS --replicate $REP --device cuda:0
  touch $R/stageA_r$REP/JOB_DONE
else
  OUT=$R/b${BUDGET}_r$REP
  SA=$R/stageA_r$REP/stageA/stage_A.pt; [ "$REP" = 0 ] && SA=$MAIN/stageA/stage_A.pt
  if [ "${REF:-0}" = 1 ] && [ ! -d $OUT/head ]; then
    mkdir -p $OUT/stream $OUT/head
    cp $MAIN/stream/best.pt $MAIN/stream/done.json $OUT/stream/
    cp $MAIN/head/controllers.pt $MAIN/head/selection.json $MAIN/head/table.json $OUT/head/
  fi
  python experiments/20261016_budget_up/budget_up.py run --data $W/data/${DS}_p$P --stagea $SA --budget $BUDGET \
    --rep $REP --fs $FS --out $OUT --device cuda:0
fi
echo "$(date '+%F %T') done"
