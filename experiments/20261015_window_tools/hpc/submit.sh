#!/bin/bash
# submit.sh TOOL [UNITS...]: window_tools.py run for TOOL (readcurrent | nanodeep | squigglenet) on the units (default all 12).
set -euo pipefail
W=/path/to/poreloom_work
S=$W/PoreLoom/experiments/20261015_window_tools/hpc/wt.sbatch
T=$1; shift
UNITS=${*:-r10_run01:0 r10_run02:0 r10_run03:0 r9:0 r9:1 r9:2 sim_g1:0 sim_g2:0 sim_g3:0 sputum_g1:0 sputum_g2:0 sputum_g3:0}
mkdir -p $W/runs/window_tools/logs
for u in $UNITS; do ds=${u%%:*} rep=${u##*:}
  [ -f $W/runs/window_tools/$ds/r$rep/$T/JOB_DONE ] && { echo "skip $T $ds r$rep"; continue; }
  squeue -u $USER -h -o %j | grep -qx "wt$T-$ds-r$rep" && { echo "queued $T $ds r$rep"; continue; }
  sbatch --parsable -J "wt$T-$ds-r$rep" -o "$W/runs/window_tools/logs/wt$T-$ds-r$rep-%j.log" --export=ALL,TOOL=$T,DS=$ds,REP=$rep $S
done
