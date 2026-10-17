#!/bin/bash
# submit.sh [method...]: every main unit (r10_run01-03 r0, sputum_g1-3 r0, sim_g1-3 r0, r9 r0-2) x method.
set -euo pipefail
W=/path/to/poreloom_work
methods=${*:-readcurrent nanodeep_upstream squigglenet}
units="r10_run01:0 r10_run02:0 r10_run03:0 sputum_g1:0 sputum_g2:0 sputum_g3:0 sim_g1:0 sim_g2:0 sim_g3:0 r9:0 r9:1 r9:2"
for u in $units; do ds=${u%%:*} rep=${u##*:}
  for m in $methods; do
    out=$W/runs/win6000/$ds/${m}_r$rep
    [ -f $out/JOB_DONE ] && { echo "skip $out"; continue; }
    mkdir -p $out
    sbatch --parsable -J "w6${m:0:2}-$ds-r$rep" -o $out/slurm-%j.log --export=ALL,METHOD=$m,DS=$ds,REP=$rep \
      $W/PoreLoom/experiments/20261016_window6000/hpc/w6.sbatch
  done
done
