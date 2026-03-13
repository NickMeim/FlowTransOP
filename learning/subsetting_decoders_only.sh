#!/bin/bash
#SBATCH -J myjob_4GPUs
#SBATCH -o generalizedTransOP_CV_train_%j.out
#SBATCH -e generalizedTransOP_CV_train_%j.err
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=23:59:59

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=gpu_lembas
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

python3 ./DecodeFromConsencusSpaceRandomPairs.py --subset_size 512 --folders HA1E_VCAP
