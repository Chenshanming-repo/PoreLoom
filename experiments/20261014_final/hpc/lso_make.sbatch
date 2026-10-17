#!/bin/bash
#SBATCH -p freecpuQ
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
# LSO fold caches for sputum group GROUP from $W/data/sputum_$GROUP (lso.py make).
set -euo pipefail
W=/path/to/poreloom_work
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate poreloom-env
cd $W/PoreLoom/experiments/20261014_final
echo "$(date '+%F %T') host $(hostname) group $GROUP"
python lso.py make --group $GROUP --splits $W/splits/sputum_$GROUP.csv --prefix $W/data/sputum_$GROUP --data $W/data
echo "$(date '+%F %T') done"
