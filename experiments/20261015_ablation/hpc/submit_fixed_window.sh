#!/bin/bash
# submit_fixed_window.sh: fixed_window.py run on the 12 ablation units (after the ablation jobs).
set -euo pipefail
W=/path/to/poreloom_work
S=$W/PoreLoom/experiments/20261015_ablation/hpc/abl.sbatch
mkdir -p $W/runs/fixed_window/logs
for u in r10_run01:0 r10_run02:0 r10_run03:0 r9:0 r9:1 r9:2 sim_g1:0 sim_g2:0 sim_g3:0 sputum_g1:0 sputum_g2:0 sputum_g3:0; do
  ds=${u%%:*} rep=${u##*:}
  [ -f $W/runs/fixed_window/$ds/r$rep/metrics.json ] && { echo "skip $ds r$rep"; continue; }
  sbatch --parsable -J "abf-$ds-r$rep" -o "$W/runs/fixed_window/logs/abf-$ds-r$rep-%j.log" --export=ALL,STAGE=fixedwin,DS=$ds,REP=$rep $S
done
