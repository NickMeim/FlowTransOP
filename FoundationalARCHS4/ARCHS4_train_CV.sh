#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 32
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=250G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH -J ARCHS4_train_fold
#SBATCH --array=0-9
#SBATCH -o logs/slurm_%x_%A_%a.out
#SBATCH -e logs/slurm_%x_%A_%a.err

module load miniforge/24.3.0-0
mamba activate nikos

FOLD="${SLURM_ARRAY_TASK_ID}"
python3 ./train_ARCHS4_fold.py --fold "${FOLD}"