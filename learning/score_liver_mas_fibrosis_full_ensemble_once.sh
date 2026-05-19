#!/bin/bash
#SBATCH -p pi_lauffen
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 30
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=72G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=36:00:00
#SBATCH -J liver_avg_score
#SBATCH -o logs/slurm_%x_%A_%a.out
#SBATCH -e logs/slurm_%x_%A_%a.err

set -euo pipefail

mkdir -p logs

module load miniforge/24.3.0-0
mamba activate nikos

OUT_DIR=${OUT_DIR:-../archs4/evaluation/liver_mas_fibrosis_full_ensemble}
ENSEMBLE_START=${ENSEMBLE_START:-0}
ENSEMBLE_END=${ENSEMBLE_END:-9}
ENSEMBLE_RANGE=${ENSEMBLE_RANGE:-${ENSEMBLE_START}-${ENSEMBLE_END}}
DECODER_SAMPLE_N=${DECODER_SAMPLE_N:-25}
DECODER_SAMPLE_TEMPERATURE=${DECODER_SAMPLE_TEMPERATURE:-1.0}
RUN_PER_ENSEMBLE=${RUN_PER_ENSEMBLE:-0}
RUN_EXPRESSION_AVERAGE=${RUN_EXPRESSION_AVERAGE:-1}
SKIP_SVM_RBF=${SKIP_SVM_RBF:-0}

EXTRA_ARGS=()
if [[ "${SKIP_SVM_RBF}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_svm_rbf)
fi

if [[ "${RUN_PER_ENSEMBLE}" == "1" ]]; then
  for ENSEMBLE_ID in $(seq "${ENSEMBLE_START}" "${ENSEMBLE_END}"); do
    echo "[$(date)] Scoring full ensemble member ${ENSEMBLE_ID}"
    python3 ./score_liver_mas_fibrosis.py \
      --model_source full_ensemble \
      --ensemble_id "${ENSEMBLE_ID}" \
      --ensemble_suffix_as_fold \
      --out_dir "${OUT_DIR}" \
      --decoder_sample_n "${DECODER_SAMPLE_N}" \
      --decoder_sample_temperature "${DECODER_SAMPLE_TEMPERATURE}" \
      "${EXTRA_ARGS[@]}"
  done
fi

if [[ "${RUN_EXPRESSION_AVERAGE}" == "1" ]]; then
  echo "[$(date)] Scoring expression-level ensemble average for members ${ENSEMBLE_RANGE}"
  python3 ./score_liver_mas_fibrosis.py \
    --model_source full_ensemble \
    --ensemble_id "${ENSEMBLE_START}" \
    --only_average_expression_ensemble \
    --average_expression_ensemble_ids "${ENSEMBLE_RANGE}" \
    --average_expression_output_suffix ensemble_expression_mean \
    --out_dir "${OUT_DIR}" \
    --decoder_sample_n "${DECODER_SAMPLE_N}" \
    --decoder_sample_temperature "${DECODER_SAMPLE_TEMPERATURE}" \
    --skip_loocv \
    --skip_ml_loocv \
    "${EXTRA_ARGS[@]}"
fi

echo "[$(date)] Done."
