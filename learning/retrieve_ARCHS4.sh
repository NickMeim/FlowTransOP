#!/bin/bash
#SBATCH -p mit_normal
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 96
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=1000G
#SBATCH --time=12:00:00
#SBATCH -J retrieve_ARCHS4

module load miniforge/24.3.0-0
mamba activate nikos


python3 ./archs4_workflow.py
