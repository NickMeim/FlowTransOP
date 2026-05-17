#!/bin/bash
#SBATCH -p pi_lauffen
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 30
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=300G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=06:00:00
#SBATCH -J liver_mas_fibrosis
#SBATCH --array=0-1
#SBATCH -o logs/slurm_%x_%A_%a.out
#SBATCH -e logs/slurm_%x_%A_%a.err

module load miniforge/24.3.0-0
mamba activate nikos

MODEL_SOURCE=${MODEL_SOURCE:-full_ensemble}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [[ "${MODEL_SOURCE}" == "full_ensemble" ]]; then
  OUT_DIR=${OUT_DIR:-../archs4/evaluation/liver_mas_fibrosis_full_ensemble}
  python3 ./score_liver_mas_fibrosis.py \
    --model_source full_ensemble \
    --ensemble_id "${TASK_ID}" \
    --ensemble_suffix_as_fold \
    --out_dir "${OUT_DIR}"
else
  python3 ./score_liver_mas_fibrosis.py \
    --model_source fold \
    --fold "${TASK_ID}"
fi
