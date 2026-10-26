#!/bin/bash
#SBATCH -A <account>
#SBATCH -q <qos>
#SBATCH --partition=<gpu partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# One training run: METHOD (poreloom | readcurrent | nanodeep_upstream | squigglenet) on DATA (dataset root) at FS Hz,
# replicate REP, output OUT. Re-submitting the same job resumes: finished stages are skipped.
set -euo pipefail
W=/path/to/poreloom_work
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate poreloom-env
export PYTHONPATH=$W/PoreLoom/src OMP_NUM_THREADS=8
export PORELOOM_DATA=$W/data PORELOOM_RESULTS=$W/runs
cd $W/PoreLoom
echo "$(date '+%F %T') host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) method $METHOD data $DATA fs $FS rep $REP"
case "$METHOD" in poreloom|readcurrent|nanodeep_upstream|squigglenet) ;; *) echo "Unknown manuscript method: $METHOD" >&2; exit 2 ;; esac
RECIPE=project; [ "$METHOD" = nanodeep_upstream ] && RECIPE=upstream
if [ "$METHOD" = poreloom ]; then
  python -m poreloom.method all --data "$DATA" --out "$OUT" --fs "$FS" --replicate "$REP" --device cuda:0
else
  python -m poreloom.method.baselines --tool "${METHOD%%_*}" --recipe "$RECIPE" --data "$DATA" --out "$OUT" --fs "$FS" --replicate "$REP" --device cuda:0
fi
touch "$OUT/JOB_DONE"
echo "$(date '+%F %T') done"
