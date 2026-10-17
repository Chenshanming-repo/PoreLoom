#!/bin/bash
# submit.sh <experiment> <dataset> <fs> <rep> <method>...   e.g. submit.sh main sputum_g1 5000 0 poreloom readcurrent
# Data: $W/data/<dataset>; output: $W/runs/<experiment>/<dataset>/<method>_r<rep>; logs next to the output.
set -euo pipefail
W=/path/to/poreloom_work
exp=$1 ds=$2 fs=$3 rep=$4; shift 4
for m in "$@"; do
  out=$W/runs/$exp/$ds/${m}_r$rep
  [ -f "$out/JOB_DONE" ] && { echo "skip $out (done)"; continue; }
  mkdir -p "$out"
  sbatch --parsable -J "${m:0:4}-$ds-r$rep" -o "$out/slurm-%j.log" \
    --export=ALL,METHOD=$m,DATA=$W/data/$ds,FS=$fs,REP=$rep,OUT=$out $W/PoreLoom/experiments/20261014_final/hpc/job.sbatch
done
