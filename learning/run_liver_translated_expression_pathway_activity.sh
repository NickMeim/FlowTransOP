#!/bin/bash
#SBATCH -p pi_lauffen
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 30
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH -J liver_hallmark_ssgsea
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

set -eo pipefail

module load miniforge/24.3.0-0
mamba activate nikos

OUT_DIR=${OUT_DIR:-../archs4/evaluation/liver_mas_fibrosis_hallmark_ssgsea}
ENSEMBLE_IDS=${ENSEMBLE_IDS:-0-9}
TOP_N=${TOP_N:-20}
BATCH_SIZE=${BATCH_SIZE:-256}

EXTRA_ARGS=()

if [[ -n "${HALLMARK_GMT:-}" ]]; then
  EXTRA_ARGS+=(--hallmark_gmt "${HALLMARK_GMT}")
fi

if [[ -n "${DEVICE:-}" ]]; then
  EXTRA_ARGS+=(--device "${DEVICE}")
fi

if [[ "${SKIP_PLOTS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_plots)
fi

python3 ./run_liver_translated_expression_pathway_activity.py \
  --ensemble_ids "${ENSEMBLE_IDS}" \
  --out_dir "${OUT_DIR}" \
  --top_n "${TOP_N}" \
  --batch_size "${BATCH_SIZE}" \
  "${EXTRA_ARGS[@]}"

echo "Finished liver translated-expression Hallmark ssGSEA analysis: ${OUT_DIR}"
