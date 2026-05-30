# FlowTransOP

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.05.27.728305-B31B1B.svg)](https://doi.org/10.64898/2026.05.27.728305)

[![ARCHS4 ensemble models DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20434738.svg)](https://doi.org/10.5281/zenodo.20434738)

GitHub repository accompanying the manuscript:

> FlowTransOP: Distributional Translation of Omics Signatures via Constrained Deep Flow Matching
>
> Nikolaos Meimetis<sup>1,*</sup>; Trong Nghia Hoang<sup>2,&dagger;</sup>; Sara Magliacane<sup>3,4,&dagger;</sup>; & Douglas A. Lauffenburger<sup>1</sup>.
>
> 1. Department of Biological Engineering, Massachusetts Institute of Technology, Cambridge, MA, 02139, USA
> 2. School of Electrical Engineering and Computer Science, Washington State University, Pullman, WA, 99164-236, USA
> 3. Informatics Institute, University of Amsterdam, Amsterdam, The Netherlands
> 4. Saarland Informatics Campus, Saarland University, Saarbr&uuml;cken, Germany
>
> &dagger; These authors contributed equally.
>
> \* Corresponding author, meimetis@mit.edu & nmeimetis97@gmail.com
>
> Manuscript: [https://doi.org/10.64898/2026.05.27.728305](https://doi.org/10.64898/2026.05.27.728305)
>
> ARCHS4 ensemble models: [https://doi.org/10.5281/zenodo.20434738](https://doi.org/10.5281/zenodo.20434738)

## Abstract

FlowTransOP is a framework for distributional translation of omics signatures
between biological domains when paired samples and one-to-one feature maps are
limited or absent. The repository contains the analyses for the manuscript,
including L1000 benchmarks, ARCHS4 mouse-human training, cross-validation, liver
evaluation, MASH case-study scoring, and plotting.

The original research scripts are in `learning/`. A lightweight installable
package scaffold is in `src/flowtransop/` for loading trained checkpoints,
running standard workflows, and translating preprocessed matrices.

**README order:** after the repository layout, this guide first shows
installation, the installable `flowtransop` package interface, and a minimal
package example. It then gives the full manuscript reproduction steps.

## Repository Layout

```text
FlowTransOP/
  README.md                         Repository-level reproduction guide
  pyproject.toml                    Installable package metadata
  src/flowtransop/                  Lightweight package and CLI
  learning/                         Python training, evaluation, scoring scripts
  preprocessing/                    L1000 preprocessing scripts
  postprocessing/                   R/Python scripts for statistics and plots
  results/                          L1000 benchmark outputs
  archs4/                           ARCHS4 raw/preprocessed data, models, evaluations
  figures/                          Supplementary/supporting figure outputs
```

## Install

Create an isolated environment with Python 3.9 or newer. The package smoke test
has been run successfully on SLURM with Python 3.10.14. Some clusters still
default to older Python versions such as 3.6; in that case, load a newer Python
module or pass an explicit Python executable before creating the environment.

For package import, CLI help, and checkpoint inference:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

For the manuscript reproduction wrappers, L1000 smoke training, ARCHS4
training/evaluation, plotting support, and scripts that import packages such as
`statsmodels`, install the reproduction extras instead:

```bash
python -m pip install -e ".[reproduce]"
```

GPU-enabled PyTorch is strongly recommended for training. The package wrappers
default to CUDA/GPU for both the model and the TRANSACT/pre-alignment backend,
while exposing separate switches for users who want one of those stages on CPU.

For plotting, install R and the R packages used by `postprocessing/`:

```r
install.packages(c(
  "tidyverse", "ggplot2", "ggpubr", "patchwork", "cowplot",
  "rstatix", "lme4", "emmeans", "ggridges", "ggsignif"
))
```

## Validate Installation

After installation, confirm that the command-line interface is available:

```bash
flowtransop --help
flowtransop run-l1000 --help
flowtransop train-archs4-fold --help
flowtransop predict --help
```

On a SLURM cluster, the repository includes a package smoke test that creates an
isolated virtual environment, installs the package, checks the CLI, and runs a
short L1000 training workflow with fast parameters:

```bash
sbatch archived/flowtransop_package_smoke_test.slurm.sh
```

The smoke test writes into `archived/flowtransop_smoke_${SLURM_JOB_ID}/`. It
requires Python >= 3.9, refuses to continue if the virtual environment fails to
activate, and installs `.[reproduce]` by default when `RUN_L1000_SMOKE=1`.
Useful overrides are:

```bash
# Install and CLI checks only
sbatch --export=ALL,RUN_L1000_SMOKE=0 archived/flowtransop_package_smoke_test.slurm.sh

# Reuse an existing smoke-test virtual environment
sbatch --export=ALL,RECREATE_ENV=0 archived/flowtransop_package_smoke_test.slurm.sh

# Use a cluster-specific Python module or executable
sbatch --export=ALL,PYTHON_MODULE=python/3.10 archived/flowtransop_package_smoke_test.slurm.sh
sbatch --export=ALL,PYTHON_MODULE='',PYTHON_BIN=/path/to/python3.10 archived/flowtransop_package_smoke_test.slurm.sh

# GPU smoke test for AutoTransOP through the package
sbatch --export=ALL,L1000_METHOD=simple-autotransop,MODEL_DEVICE=cuda,TRANSACT_BACKEND=gpu,TRANSACT_DEVICE=cuda archived/flowtransop_package_smoke_test.slurm.sh

# GPU smoke test for consensus-space decoders through the package
sbatch --export=ALL,L1000_METHOD=consensus-decoders,MODEL_DEVICE=cuda,TRANSACT_BACKEND=gpu,TRANSACT_DEVICE=cuda archived/flowtransop_package_smoke_test.slurm.sh

# Hybrid FlowTransOP smoke test with a non-default pair/similarity aggregation
sbatch --export=ALL,L1000_METHOD=hybrid-flowtransop,HYBRID_AGGREGATION=mean,MODEL_DEVICE=cuda,TRANSACT_BACKEND=gpu,TRANSACT_DEVICE=cuda archived/flowtransop_package_smoke_test.slurm.sh
```

## Package CLI

After `pip install -e .`, inference commands and help pages can be called with
`flowtransop`. For commands that delegate to the original manuscript training or
evaluation scripts, use `pip install -e ".[reproduce]"`. The package exposes
model and TRANSACT/pre-alignment device choices separately:

The L1000 wrapper is installed by default with the package and includes the
three main manuscript options:

| Method argument | Workflow |
| --- | --- |
| `--method hybrid-flowtransop` | Standard hybrid FlowTransOP using known paired-condition indicators plus TRANSACT pre-aligned similarity. The default aggregation is `--hybrid-aggregation max`, the manuscript-selected setting. |
| `--method simple-autotransop` or `--method autotransop` | AutoTransOP/CPA-style baseline. The CLI prints a hyperparameter-sensitivity warning when this is run. |
| `--method consensus-decoders` | Consensus-space decoder baseline. |

```bash
flowtransop train-archs4-fold --repo-root . --fold 0 --direction h2m \
  --model-device cuda --transact-backend gpu --transact-device cuda
flowtransop train-archs4-fold --repo-root . --fold 0 --direction m2h \
  --model-device cuda --transact-backend gpu --transact-device cuda
flowtransop train-archs4-ensemble --repo-root . --ensemble-id 0 --fold 0 \
  --model-device cuda --transact-backend gpu --transact-device cuda
flowtransop evaluate-archs4-fold --repo-root . --fold 0 --include-liver
```

L1000 method wrappers are also available:

```bash
flowtransop run-l1000 --repo-root . --method consensus-decoders
flowtransop run-l1000 --repo-root . --method hybrid-flowtransop
flowtransop run-l1000 --repo-root . --method simple-autotransop
```

For the standard hybrid, exact paired-condition matches are encoded in `C` and
approximate relationships from the pre-aligned TRANSACT space are encoded in
`C_pre`. By default, FlowTransOP combines them elementwise with
`--hybrid-aggregation max`, preserving exact known-pair signal while allowing
TRANSACT similarity to guide non-identical or partially matched samples:

```bash
flowtransop run-l1000 --repo-root . --method hybrid-flowtransop --hybrid-aggregation max
flowtransop run-l1000 --repo-root . --method hybrid-flowtransop --hybrid-aggregation mean
flowtransop run-l1000 --repo-root . --method hybrid-flowtransop --hybrid-aggregation sum
```

Use the pretrained full ARCHS4 ensemble weights from a local, gitignored
`archs4/` folder. The ARCHS4 ensemble model files are deposited on Zenodo:
[![ARCHS4 ensemble models DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20434738.svg)](https://doi.org/10.5281/zenodo.20434738)

```bash
flowtransop predict-archs4-ensemble \
  --archs4-dir archs4 \
  --ensemble-ids 0-9 \
  --direction m2h \
  --input-npy archs4/preprocessed/mouse_test_X.npy \
  --output-npy archs4/evaluation/example_ensemble_m2h_prediction.npy
```

This expects local checkpoints such as:

```text
archs4/models/full_ensemble_0_normal.pt
...
archs4/models/full_ensemble_9_normal.pt
```

The package also exposes a fine-tuning entry point for users who have the
preprocessed ARCHS4 matrices and want to continue from one pretrained ensemble
member:

```bash
flowtransop finetune-archs4-ensemble \
  --repo-root . \
  --archs4-dir archs4 \
  --ensemble-id 0 \
  --epochs 5 \
  --output-model-dir archs4/models_finetuned
```

Fine-tuned checkpoints are written to a separate directory by default
(`archs4/models_finetuned`) so the original pretrained weights are not
overwritten.

To run model code on GPU but expose CPU TRANSACT/pre-alignment settings:

```bash
flowtransop train-archs4-fold --repo-root . --fold 0 --direction h2m \
  --model-device cuda --transact-backend cpu --transact-device cpu
```

Python users can choose the TRANSACT backend independently as well:

```python
from flowtransop import RuntimeBackends, load_transact_backend

backends = RuntimeBackends(model_device="cuda", transact_backend="cpu", transact_device="cpu")
transact = load_transact_backend(repo_root=".", backends=backends)
Z_source, Z_target, tau, model = transact.align(X_source, X_target)
```

Python users can also load the pretrained ARCHS4 ensemble directly:

```python
from flowtransop import load_archs4_ensemble

ensemble = load_archs4_ensemble(archs4_dir="archs4", ensemble_ids="0-9")
translated = ensemble.translate(mouse_expression, direction="m2h")
```

Translate a preprocessed matrix with a saved checkpoint:

```bash
flowtransop predict \
  --normal-checkpoint archs4/models/full_ensemble_0_normal.pt \
  --direction h2m \
  --input-npy path/to/preprocessed_human_samples.npy \
  --output-npy translated_mouse_expression.npy
```

Inputs must already be normalized and feature-ordered like the matrices used
for training.

## Minimal End-to-End Package Example

From the repository root:

```bash
python -m pip install -e ".[reproduce]"

flowtransop train-archs4-fold --repo-root . --fold 0 --direction h2m \
  --model-device cuda --transact-backend gpu --transact-device cuda
flowtransop train-archs4-fold --repo-root . --fold 0 --direction m2h \
  --model-device cuda --transact-backend gpu --transact-device cuda

flowtransop evaluate-archs4-fold --repo-root . --fold 0 --include-liver

flowtransop predict \
  --normal-checkpoint archs4/models/fold_0_normal.pt \
  --m2h-checkpoint archs4/models/fold_0_normal_m2h.pt \
  --direction m2h \
  --input-npy archs4/preprocessed/mouse_test_X.npy \
  --output-npy archs4/evaluation/example_m2h_prediction.npy
```

## Reproducing the Study

Most scripts assume they are run from `learning/`, because paths are written
relative to that folder, e.g. `../archs4`, `../results`, and `../postprocessing`.

### 1. L1000 Benchmark Data

The L1000 experiments in the active manuscript pipeline use
AutoTransOP-compatible processed L1000 matrices, cell-line task definitions, and
fold splits under:

```text
preprocessing/preprocessed_data/CellPairs/
preprocessing/preprocessed_data/SameCellimputationModel/
preprocessing/preprocessed_data/SameCellimputationModel/bracketed_difficulty/
```

`preprocessing/preProcessL1000DrugData.R` creates older pair, triplet, and
quadruplet split files, but those exact artifacts are not referenced by the
current learning or postprocessing scripts. Treat it as a legacy/optional helper
rather than a required manuscript reproduction step. See
`preprocessing/readme.md` for the details.

### 2. L1000 Benchmark Models

Run these from `learning/`. The manuscript-scale jobs were run through SLURM,
and the `.sh` files in `learning/` are SLURM submission scripts. Use `sbatch`
for reproducing the study; direct Python commands are mainly useful for small
debugging runs or when adapting the workflow to another scheduler.
For compact commands shown with `sbatch --wrap`, use the same resource requests
and environment activation used in the checked-in SLURM wrappers on your cluster.

```bash
cd learning

# Shared-feature cell-line benchmark
sbatch cell_pairs_benchmark.sh

# Optional Supplementary Figure S1 ablation:
# shared-feature FlowTransOP with structural guidance disabled
sbatch supplementary_figure_s1_no_structural_guidance.sh

# Optional reverse direction for the same S1 ablation if flow21 outputs are missing
sbatch supplementary_figure_s1_no_structural_guidance_reverse.sh

# Low-pair and extremely-low-pair benchmarks
sbatch low_percentage_of_pairs.sh
sbatch extremely_low_percentage_of_pairs.sh
sbatch pairedFlow_low_percentage_of_pairs.sh
sbatch pairedFlow_low_percentage_of_pairs_extreme.sh

# Distinct-feature benchmarks and decoder-only baselines
sbatch OneCell_differentInputs_benchmark.sh
sbatch decoders_only_imputedGenes.sh
sbatch subsetting_decoders_only.sh
```

Outputs are written under `results/` and summarized by the plotting scripts in
`postprocessing/`. The S1 ablation writes
`results/FlowMatch_no_structural_guidance/`; compare it against the main
`results/AutoTransOP_CellPairs/` run when assembling the no-structural-guidance
supplementary panel.

The installable package also exposes a convenience wrapper for the main L1000
choices discussed in the manuscript:

```bash
flowtransop run-l1000 --repo-root . --method consensus-decoders
flowtransop run-l1000 --repo-root . --method hybrid-flowtransop
flowtransop run-l1000 --repo-root . --method simple-autotransop
```

Use `--method consensus-decoders` when you want the consensus-space decoder
baseline, `--method hybrid-flowtransop` when you want the pair-and-similarity
FlowTransOP variant, and `--method simple-autotransop` when you want the
AutoTransOP/CPA-style baseline. Extra arguments are passed through to the
selected script, for example `--output_dir`, `--epochs`, `--folders`, or
checkpoint/log options supported by that script.

The standard hybrid uses `--hybrid-aggregation max` by default. This combines
the exact known-pair matrix and the TRANSACT-derived approximate similarity
matrix by taking the elementwise maximum. Users can instead pass
`--hybrid-aggregation mean` or `--hybrid-aggregation sum` when they want to test
average or additive pair/similarity aggregation.

**Important AutoTransOP note:** AutoTransOP hyperparameters are very important
and the method can be highly sensitive. Whether to use mutual information,
cosine distance, Euclidean distance, and/or prior/adversarial discriminators as
proposed in the original publication is a modeling choice that users must
customly re-adjust for their own data, paired-sample regime, and feature space.
The checked-in defaults should not be treated as universally optimal.

### 3. ARCHS4 Download, Splits, and Preprocessing

Run from `learning/` using the SLURM wrappers:

```bash
sbatch retrieve_ARCHS4.sh
sbatch preprocess_ARCHS4.sh
sbatch mouse_preprocess.sh
```

This creates or expects:

```text
archs4/human_gene_v2.latest.h5
archs4/mouse_gene_v2.latest.h5
archs4/splits/
archs4/preprocessed/
```

For resumable/preemptable mouse preprocessing, use
`sbatch mouse_preprocess_preemptable.sh`.

### 4. ARCHS4 Cross-Validation Training

For each fold, train human-to-mouse first, then mouse-to-human. The manuscript
models were trained through SLURM:

```bash
sbatch --array=0-9 ARCHS4_train_CV.sh
```

The reverse-direction script can be submitted analogously if it is not included
in the local scheduler wrapper:

```bash
sbatch --array=0-9 --wrap='python train_ARCHS4_fold_m2h.py --fold ${SLURM_ARRAY_TASK_ID}'
```

The CV checkpoints are written to `archs4/models/`:

```text
fold_{fold}_normal.pt
fold_{fold}_permuted.pt
fold_{fold}_normal_m2h.pt
fold_{fold}_permuted_m2h.pt
```

### 5. ARCHS4 Evaluation

Run per fold through SLURM:

```bash
sbatch --array=0-9 evaluate_translation.sh
sbatch --array=0-9 --wrap='python evaluate_expression_mmd_archs4.py --fold ${SLURM_ARRAY_TASK_ID}'
sbatch --array=0-9 --wrap='python evaluate_liver.py --fold ${SLURM_ARRAY_TASK_ID}'
```

These write CSV outputs to `archs4/evaluation/`.

### 6. Full ARCHS4 Ensemble and MASH Scoring

The final MASH case study uses full-data ensemble models. Submit the ensemble
array through SLURM:

```bash
sbatch --array=0-9 ARCHS4_train_full_ensemble.sh
```

Then score the liver MASH/fibrosis studies:

```bash
sbatch score_liver_mas_fibrosis_final_expression_mean.sh
```

Outputs are written to:

```text
archs4/evaluation/liver_mas_fibrosis_final_expression_mean/
```

This script imports shared scoring helpers from `learning/score_liver_mas_fibrosis.py`;
keep that helper alongside the final scoring script when recreating the case
study.

### 7. Plotting

Run from `postprocessing/`:

```bash
Rscript evaluate5folds.R
Rscript LowPairsPerformance.R
Rscript DifferentInputsPerformanceBracketed.R
Rscript GPU_vs_CPU_implementation.R

Rscript plot_archs4_evaluation.R
Rscript plot_archs4_liver_evaluation.R
Rscript plot_liver_mas_fibrosis_final_expression_mean.R
```

ARCHS4 figures are written under:

```text
archs4/evaluation/figures_flowtransop/
archs4/evaluation/figures_liver/
archs4/evaluation/liver_mas_fibrosis_final_expression_mean/figures/
```

## Notes for Reuse

- The original scripts remain the source of truth for exact manuscript analyses.
- The package scaffold provides stable loading and inference around the saved
  checkpoint format.
- Training expects large memory and GPU resources for ARCHS4-scale data.
- Random/permuted baselines are important for interpreting unpaired translation
  performance and should be retained when benchmarking new models.
