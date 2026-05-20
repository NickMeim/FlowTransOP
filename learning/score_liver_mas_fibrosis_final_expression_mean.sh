#!/bin/bash
#SBATCH -p pi_lauffen
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 30
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=72G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=12:00:00
#SBATCH -J liver_final_plsr
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

set -eo pipefail

mkdir -p logs

module load miniforge/24.3.0-0
mamba activate nikos

OUT_DIR=${OUT_DIR:-../archs4/evaluation/liver_mas_fibrosis_final_expression_mean}
ENSEMBLE_IDS=${ENSEMBLE_IDS:-0-9}

python3 ./score_liver_mas_fibrosis_final_expression_mean.py \
  --ensemble_ids "${ENSEMBLE_IDS}" \
  --out_dir "${OUT_DIR}"

echo "[$(date)] Done."
