#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition batch
#SBATCH --open-mode=append
#SBATCH --output=slurm_joint_out.txt
#SBATCH --error=slurm_joint_err.txt

# Adapt the module / conda activation and the working directory to your cluster.
# source $VSC_DATA/miniforge3/etc/profile.d/conda.sh
# conda activate pdaep
# cd $VSC_DATA/PD-Adverse-Event-Prediction

python -u run_joint.py "$@"
