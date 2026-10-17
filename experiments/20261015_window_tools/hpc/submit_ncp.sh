#!/bin/bash
# submit_ncp.sh: native_cp.py on the 12 units. Skips units done or queued.
set -euo pipefail
W=/path/to/poreloom_work
S=$W/PoreLoom/experiments/20261015_window_tools/hpc/ncp.sbatch
mkdir -p $W/runs/native_cp/logs
for u in r10_run01:0 r10_run02:0 r10_run03:0 r9:0 r9:1 r9:2 sim_g1:0 sim_g2:0 sim_g3:0 sputum_g1:0 sputum_g2:0 sputum_g3:0; do
  ds=${u%%:*} rep=${u##*:}
  [ -f $W/runs/native_cp/$ds/r$rep/JOB_DONE ] && { echo "skip $ds r$rep"; continue; }
  squeue -u $USER -h -o %j | grep -qx "ncp-$ds-r$rep" && { echo "queued $ds r$rep"; continue; }
  sbatch --parsable -J "ncp-$ds-r$rep" -o "$W/runs/native_cp/logs/ncp-$ds-r$rep-%j.log" --export=ALL,DS=$ds,REP=$rep $S
done
