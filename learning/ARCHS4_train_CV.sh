#!/bin/bash
#SBATCH -p pi_lauffen
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 30
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=300G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=48:00:00
#SBATCH -J ARCHS4_train_fold
#SBATCH --array=0-1
#SBATCH -o logs/slurm_%x_%A_%a.out
#SBATCH -e logs/slurm_%x_%A_%a.err

module load miniforge/24.3.0-0
mamba activate nikos

python3 ./train_ARCHS4_fold.py --fold ${SLURM_ARRAY_TASK_ID}
