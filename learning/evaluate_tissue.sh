#!/bin/bash
#SBATCH -p mit_normal
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 64
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH -J ARCHS4_tissue_eval
#SBATCH --array=0-9
#SBATCH -o logs/slurm_%x_%A_%a.out
#SBATCH -e logs/slurm_%x_%A_%a.err

module load miniforge/24.3.0-0
mamba activate nikos

python3 ./evaluate_tissue.py --fold ${SLURM_ARRAY_TASK_ID}
