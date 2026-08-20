#!/usr/bin/env bash
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition batch
#SBATCH --open-mode=append
#SBATCH --output=slurm_within_out.txt
#SBATCH --error=slurm_within_err.txt

# Adapt the module / conda activation and the working directory to your cluster.
# source $VSC_DATA/miniforge3/etc/profile.d/conda.sh
# conda activate pdaep
# cd $VSC_DATA/PD-Adverse-Event-Prediction

python -u run_within.py "$@"
