#!/bin/bash
# submit.sh <ds> <fs> <prefix> "<budgets>" <tag> [stagea] [ref6000]
#   r10:    submit.sh r10_run01 4000 27000 "6000 9000 12000 15000 18000" bu
#   sputum: submit.sh sputum_g1 5000 16500 "6000 9000 12000" sp ref6000
# Stage A r1, r2 (default 9000 cache), then budgets x replicates 0-2 (r1/r2 after their stage A). `stagea`: stage A only.
# `ref6000`: B 6000 r0 = the main run (S1 + head copied, TEST replayed). Skips finished and queued jobs.
set -euo pipefail
W=/path/to/poreloom_work
S=$W/PoreLoom/experiments/20261016_budget_up/hpc/bu.sbatch
DS=$1 FS=$2 P=$3 BUDGETS=$4 TAG=$5; shift 5
ONLY_A=0 REF6000=0; for x in "$@"; do [ "$x" = stagea ] && ONLY_A=1; [ "$x" = ref6000 ] && REF6000=1; done
R=$W/runs/budget_up/$DS; mkdir -p $R/logs
EXP=ALL,DS=$DS,FS=$FS,P=$P
declare -A dep
for rep in 1 2; do
  if [ -f $R/stageA_r$rep/JOB_DONE ]; then dep[$rep]=""; continue; fi
  q=$(squeue -u $USER -h -n ${TAG}a-r$rep -o %i | head -1)
  if [ -n "$q" ]; then dep[$rep]="--dependency=afterok:$q"; echo "stageA r$rep queued $q"; continue; fi
  id=$(sbatch --parsable -J ${TAG}a-r$rep -o $R/logs/stagea_r$rep-%j.log --export=$EXP,STAGE=stagea,REP=$rep $S)
  dep[$rep]="--dependency=afterok:$id"; echo "stageA r$rep $id"
done
dep[0]=""
[ $ONLY_A = 1 ] && exit 0
for rep in 0 1 2; do for b in $BUDGETS; do
  [ -f $R/b${b}_r$rep/JOB_DONE ] && { echo "skip b$b r$rep"; continue; }
  [ -n "$(squeue -u $USER -h -n $TAG$((b/1000))-r$rep -o %i)" ] && { echo "queued b$b r$rep"; continue; }
  ref=0; [ $REF6000 = 1 ] && [ $b = 6000 ] && [ $rep = 0 ] && ref=1
  id=$(sbatch --parsable ${dep[$rep]} -J $TAG$((b/1000))-r$rep -o $R/logs/b${b}_r$rep-%j.log \
    --export=$EXP,STAGE=run,BUDGET=$b,REP=$rep,REF=$ref $S)
  echo "b$b r$rep $id"
done; done
