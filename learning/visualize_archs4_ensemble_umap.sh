#!/bin/bash
#SBATCH -p pi_lauffen
#SBATCH --mail-user=meimetis@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -c 28
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=48:00:00
#SBATCH -J archs4_ens_umap
#SBATCH -o logs/slurm_%x_%j.out
#SBATCH -e logs/slurm_%x_%j.err

set -eo pipefail

module load miniforge/24.3.0-0
mamba activate nikos

OUT_DIR=${OUT_DIR:-../archs4/evaluation/archs4_full_ensemble_umap}
ENSEMBLE_IDS=${ENSEMBLE_IDS:-0-9}
N_HUMAN=${N_HUMAN:-all}
N_MOUSE=${N_MOUSE:-all}
TARGET_REFERENCE=${TARGET_REFERENCE:-raw}
TOP_VARIABLE_GENES=${TOP_VARIABLE_GENES:-5000}
PCA_COMPONENTS=${PCA_COMPONENTS:-50}
UMAP_NEIGHBORS=${UMAP_NEIGHBORS:-50}
UMAP_MIN_DIST=${UMAP_MIN_DIST:-0.25}
BATCH_SIZE=${BATCH_SIZE:-12192}
FLOW_STEPS=${FLOW_STEPS:-10}
SEED=${SEED:-20260522}
DEVICE=${DEVICE:-auto}

EXTRA_ARGS=()

if [[ "${INCLUDE_TEST:-1}" == "0" ]]; then
  EXTRA_ARGS+=(--no-include_test)
fi

if [[ "${SAVE_EMBEDDING_MATRIX:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--save_embedding_matrix)
fi

if [[ "${SKIP_H2M:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_h2m)
fi

if [[ "${SKIP_M2H:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_m2h)
fi

python3 ./visualize_archs4_ensemble_umap.py \
  --out_dir "${OUT_DIR}" \
  --ensemble_ids "${ENSEMBLE_IDS}" \
  --n_human "${N_HUMAN}" \
  --n_mouse "${N_MOUSE}" \
  --target_reference "${TARGET_REFERENCE}" \
  --top_variable_genes "${TOP_VARIABLE_GENES}" \
  --pca_components "${PCA_COMPONENTS}" \
  --umap_neighbors "${UMAP_NEIGHBORS}" \
  --umap_min_dist "${UMAP_MIN_DIST}" \
  --batch_size "${BATCH_SIZE}" \
  --flow_steps "${FLOW_STEPS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"

echo "Finished ARCHS4 ensemble UMAP visualization: ${OUT_DIR}"
