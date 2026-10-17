#!/bin/bash
# submit.sh: every ablation job (12 steps + 12 scratch_s1 + 4 window). Skips parts already done.
set -euo pipefail
W=/path/to/poreloom_work
S=$W/PoreLoom/experiments/20261015_ablation/hpc/abl.sbatch
UNITS="r10_run01:0 r10_run02:0 r10_run03:0 r9:0 r9:1 r9:2 sim_g1:0 sim_g2:0 sim_g3:0 sputum_g1:0 sputum_g2:0 sputum_g3:0"
sub() {  # name done-file exports
  [ -f "$2" ] && { echo "skip $1"; return; }
  mkdir -p $W/runs/ablation/logs
  sbatch --parsable -J "$1" -o "$W/runs/ablation/logs/$1-%j.log" --export=ALL,$3 $S
}
for u in $UNITS; do ds=${u%%:*} rep=${u##*:}
  sub "abs-$ds-r$rep" $W/runs/ablation/$ds/r$rep/STEPS_DONE "STAGE=steps,DS=$ds,REP=$rep"
  sub "abz-$ds-r$rep" $W/runs/ablation/$ds/r$rep/scratch_s1/JOB_DONE "STAGE=scratch,DS=$ds,REP=$rep"
done
for ds in sputum_g1 r10_run01; do for win in 1000 3000; do
  sub "abw-$ds-$win" $W/runs/ablation/$ds/r0/window_$win/JOB_DONE "STAGE=window,DS=$ds,REP=0,WIN=$win"
done; done
