#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 32
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=380G
#SBATCH --time=20:00:00
#SBATCH -J preprocess_ARCHS4_mouse
#SBATCH -o slurm_%j.out
#SBATCH -e slurm_%j.err
#SBATCH --exclude=node3401

module load miniforge/24.3.0-0
mamba activate nikos

python3 ./preprocess_archs4_mouse.py
