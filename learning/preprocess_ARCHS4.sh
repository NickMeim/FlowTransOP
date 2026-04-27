#!/bin/bash
#SBATCH -p mit_normal
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 96
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=380G
#SBATCH --time=12:00:00
#SBATCH -J preprocess_ARCHS4
#SBATCH -o slurm_%j.out
#SBATCH -e slurm_%j.err

module load miniforge/24.3.0-0
mamba activate nikos


python3 ./preprocess_archs4.py
