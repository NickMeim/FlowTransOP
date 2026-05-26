# FlowTransOP Learning Scripts

This folder contains the Python and shell scripts used for model training,
evaluation, and disease-score analysis. Most paths in these scripts are relative
to this folder, so run commands from `learning/` unless noted otherwise.

The manuscript-scale workflows were run through SLURM. The `.sh` files here are
SLURM submission scripts, so the reproduction examples use `sbatch`; direct
Python calls are mainly useful for debugging or porting to another scheduler.
For compact commands shown with `sbatch --wrap`, use the same resource requests
and environment activation used in the checked-in SLURM wrappers on your cluster.

## Important Data Note

The active L1000 manuscript workflow uses AutoTransOP-compatible processed files
under `../preprocessing/preprocessed_data/CellPairs/` and
`../preprocessing/preprocessed_data/SameCellimputationModel/`.

`../preprocessing/preProcessL1000DrugData.R` creates older pair, triplet, and
quadruplet split files. A repository-wide search did not find those exact
outputs being consumed by the current learning or postprocessing scripts, so it
should be treated as legacy/optional unless you intentionally want to reproduce
that older split format.

## Main Script Groups

### L1000 Benchmarks

Shared-feature, paired or low-pair benchmarks:

```text
AutoTransOP_Pretrain_FlowMatch.py
AutoTransOP_Pretrain_FlowMatch_withPairs.py
AutoTransOP_Pretrain_FlowMatch_lowPairsPercentage.py
AutoTransOP_Pretrain_FlowMatchPaired_lowPairsPercentage.py
AutoTransOP_Pretrain_FlowMatch_lowPairsPercentageExtreme.py
AutoTransOP_Pretrain_FlowMatch_lowPairsPercentageExtreme_withPairs.py
FlowMatch_lowPairsPercentage_PairsAndSimilarity.py
FlowMatch_lowPairsPercentageExtreme_PairsAndSimilarity*.py
AutoTransOP_lowPairsPercentageExtreme.py
```

Distinct-feature and decoder-only benchmarks:

```text
AutoTransOP_Pretrain_FlowMatch_differentInputs.py
AutoTransOP_Pretrain_FlowMatch_differentInputs_bracketed.py
DecodeFromConsencusSpace*.py
```

TRANSACT GPU/CPU validation:

```text
InitialAligner_GPUvsCPU_*.py
```

Convenience wrappers:

```text
cell_pairs_benchmark.sh
low_percentage_of_pairs.sh
extremely_low_percentage_of_pairs.sh
pairedFlow_low_percentage_of_pairs.sh
pairedFlow_low_percentage_of_pairs_extreme.sh
OneCell_differentInputs_benchmark.sh
decoders_only_imputedGenes.sh
subsetting_decoders_only.sh
```

These scripts write outputs to `../results/`.

### ARCHS4 Workflow

Prepare ARCHS4 splits and preprocessed matrices:

```bash
sbatch retrieve_ARCHS4.sh
sbatch preprocess_ARCHS4.sh
sbatch mouse_preprocess.sh
```

Expected outputs:

```text
../archs4/splits/
../archs4/preprocessed/human_X.npy
../archs4/preprocessed/mouse_X.npy
../archs4/preprocessed/human_test_X.npy
../archs4/preprocessed/mouse_test_X.npy
../archs4/preprocessed/*_fold{k}_train_idx.npy
../archs4/preprocessed/*_fold{k}_val_idx.npy
```

Train ARCHS4 cross-validation models:

```bash
sbatch --array=0-9 ARCHS4_train_CV.sh
sbatch --array=0-9 --wrap='python train_ARCHS4_fold_m2h.py --fold ${SLURM_ARRAY_TASK_ID}'
```

The first command trains encoders, decoders, human-to-mouse flow, and permuted
controls. The second command loads those saved encoders/decoders and trains the
mouse-to-human flow.

If the checked-in SLURM array range is narrower than the full experiment, update
the scheduler header or command-line array range before submitting.

Train full-data ensemble models for the MASH case study:

```bash
sbatch --array=0-9 ARCHS4_train_full_ensemble.sh
```

Model outputs:

```text
../archs4/models/fold_{fold}_normal.pt
../archs4/models/fold_{fold}_permuted.pt
../archs4/models/fold_{fold}_normal_m2h.pt
../archs4/models/fold_{fold}_permuted_m2h.pt
../archs4/models/full_ensemble_{id}_normal.pt
../archs4/models/full_ensemble_{id}_normal_m2h.pt
```

### ARCHS4 Evaluation

Run per fold:

```bash
sbatch --array=0-9 evaluate_translation.sh
sbatch --array=0-9 --wrap='python evaluate_expression_mmd_archs4.py --fold ${SLURM_ARRAY_TASK_ID}'
sbatch --array=0-9 --wrap='python evaluate_liver.py --fold ${SLURM_ARRAY_TASK_ID}'
```

`evaluate_translation.py` produces cycle consistency, orthologue preservation,
and latent MMD metrics. `evaluate_expression_mmd_archs4.py` adds expression
space MMD. `evaluate_liver.py` evaluates held-out liver reconstruction, cycle
consistency, orthologues, MMD, and centroid metrics.

### MASH/Fibrosis Scoring

After training full ensemble models:

```bash
sbatch score_liver_mas_fibrosis_final_expression_mean.sh
```

The script averages translated expression across ensemble members, trains human
PLSR models for MAS/NAS score and fibrosis stage, writes LOOCV performance, and
scores the selected mouse treatment studies.

Outputs:

```text
../archs4/evaluation/liver_mas_fibrosis_final_expression_mean/
```

`score_liver_mas_fibrosis_final_expression_mean.py` imports shared helper code
from `score_liver_mas_fibrosis.py`; keep that helper in this folder when running
the scoring workflow.

## Simple Script List

In the style of the companion OmicTranslationBenchmark repository, this section
lists the learning scripts first as a plain file-by-file description.

1. `archs4_workflow.py`: Script to retrieve and organize ARCHS4 human/mouse data, define liver held-out samples, and create fold split metadata.
2. `preprocess_archs4.py`: Script to convert ARCHS4 expression data into sample-major matrices and fold index arrays for training.
3. `preprocess_archs4_mouse.py`: Script to preprocess large ARCHS4 mouse expression matrices, including resumable/preemptable runs.
4. `train_ARCHS4_fold.py`: Script to train fold-level human/mouse autoencoders, the human-to-mouse flow, and permuted controls.
5. `train_ARCHS4_fold_m2h.py`: Script to train the reverse mouse-to-human flow from saved fold-level encoders and decoders.
6. `train_ARCHS4_full_ensemble.py`: Script to train one full-data ARCHS4 ensemble member for the MASH/fibrosis case study.
7. `evaluate_translation.py`: Script to evaluate ARCHS4 cycle consistency, orthologue preservation, and latent MMD metrics.
8. `evaluate_expression_mmd_archs4.py`: Script to evaluate expression-space MMD for ARCHS4 translated samples.
9. `evaluate_liver.py`: Script to evaluate held-out liver reconstruction, cycle, orthologue, MMD, and centroid metrics.
10. `score_liver_mas_fibrosis_final_expression_mean.py`: Script to average ensemble translated expression and score selected mouse MASH/fibrosis studies with human PLSR models.
11. `AutoTransOP_Pretrain_FlowMatch.py`: Script to train the main unpaired FlowTransOP L1000 shared-feature cell-line benchmark.
12. `AutoTransOP_Pretrain_FlowMatch_withPairs.py`: Script to train the paired-constrained FlowTransOP comparison for cell-line pairs.
13. `AutoTransOP_Pretrain_FlowMatch_withSTRUCTURE.py`: Script to train the STRUCTURE-initialized comparison workflow.
14. `DecodeFromConsencusSpace.py`: Script to train/evaluate the consensus-space decoder baseline for shared-feature L1000 tasks.
15. `DecodeFromConsencusSpaceRandomPairs.py`: Script to run consensus-space decoder controls with random/unpaired consensus construction.
16. `AutoTransOP_Pretrain_FlowMatch_differentInputs.py`: Script to train FlowTransOP when the two domains use different feature subsets.
17. `DecodeFromConsencusSpace_diffenetInputs.py`: Script to train/evaluate decoder-only baselines for different-input tasks.
18. `AutoTransOP_Pretrain_FlowMatch_differentInputs_bracketed.py`: Script to train FlowTransOP across bracketed feature-correlation difficulty levels.
19. `DecodeFromConsencusSpace_diffenetInputs_bracketed.py`: Script to train/evaluate decoder-only baselines across bracketed difficulty levels.
20. `AutoTransOP_Pretrain_FlowMatch_lowPairsPercentage.py`: Script to train FlowTransOP in the low-pair A375/HT29 benchmark.
21. `AutoTransOP_Pretrain_FlowMatchPaired_lowPairsPercentage.py`: Script to train paired FlowTransOP in the low-pair A375/HT29 benchmark.
22. `AutoTransOP_Pretrain_FlowMatch_lowPairsPercentageExtreme.py`: Script to train FlowTransOP in the extreme few-pair benchmark.
23. `AutoTransOP_Pretrain_FlowMatch_lowPairsPercentageExtreme_withPairs.py`: Script to train paired FlowTransOP in the extreme few-pair benchmark.
24. `AutoTransOP_lowPairsPercentageExtreme.py`: Script to train the AutoTransOP baseline under the extreme few-pair design.
25. `FlowMatch_lowPairsPercentage_PairsAndSimilarity.py`: Script to train a hybrid low-pair FlowTransOP variant using pair and similarity information.
26. `FlowMatch_lowPairsPercentageExtreme_PairsAndSimilarity.py`: Script to train the extreme few-pair pair-and-similarity variant.
27. `FlowMatch_lowPairsPercentageExtreme_PairsAndSimilarity_meanAgg.py`: Script to train/evaluate the mean-aggregation pair-and-similarity variant.
28. `FlowMatch_lowPairsPercentageExtreme_PairsAndSimilarity_sumAgg.py`: Script to train/evaluate the sum-aggregation pair-and-similarity variant.
29. `InitialAligner_GPUvsCPU_random_data.py`: Script to compare GPU and CPU TRANSACT/pre-alignment implementations on random data.
30. `InitialAligner_GPUvsCPU_celline_pairs.py`: Script to compare GPU and CPU TRANSACT/pre-alignment implementations on cell-line pair data.
31. `InitialAligner_GPUvsCPU_celline_pairs_random_subsampling.py`: Script to compare GPU/CPU TRANSACT under random subsampling of real cell-line data.
32. `InitialAligner_GPUvsCPU_sameCell_diffInput.py`: Script to compare GPU/CPU TRANSACT for same-cell, different-feature inputs.
33. `models.py`: Contains classes used to define the FlowTransOP neural-network modules.
34. `models_autotransop.py`: Contains classes used to define AutoTransOP-style baseline modules.
35. `trainingUtils.py`: Contains functions used to train FlowTransOP models.
36. `trainingUtils_autotransop.py`: Contains functions used to train AutoTransOP-style baseline models.
37. `transact_utility_gpu.py`: Contains GPU-oriented TRANSACT/pre-alignment functions.
38. `utility.py`: Contains general utilities, including CPU-side alignment/helper routines.
39. `evaluationUtils.py`: Contains functions for model evaluation and metric calculation.

## Script-by-Script Map

This table traces each learning script to the result family it generates and,
where possible, the plotting script that consumes the output.

| Script | What it does | Main outputs or downstream use |
| --- | --- | --- |
| `archs4_workflow.py` | Retrieves/organizes ARCHS4 human and mouse data, defines held-out liver samples, and creates fold split metadata. | `../archs4/splits/`; inputs for ARCHS4 preprocessing and training. |
| `preprocess_archs4.py` | Converts ARCHS4 human and mouse expression into sample-major numpy matrices and fold index arrays. | `../archs4/preprocessed/`; required by ARCHS4 train/evaluate/score scripts. |
| `preprocess_archs4_mouse.py` | Mouse-focused/resumable ARCHS4 preprocessing path for large mouse matrices. | `../archs4/preprocessed/mouse_*`; used with `mouse_preprocess*.sh`. |
| `train_ARCHS4_fold.py` | Trains fold-level human and mouse autoencoders, the human-to-mouse flow, and permuted controls. | `../archs4/models/fold_*_normal.pt`, `fold_*_permuted.pt`; required before `train_ARCHS4_fold_m2h.py` and evaluation. |
| `train_ARCHS4_fold_m2h.py` | Trains the reverse mouse-to-human flow using the fold encoders/decoders trained above. | `../archs4/models/fold_*_normal_m2h.pt`, `fold_*_permuted_m2h.pt`; consumed by ARCHS4 evaluation. |
| `train_ARCHS4_full_ensemble.py` | Trains full-data ensemble models for final translated-expression averaging. | `../archs4/models/full_ensemble_*`; consumed by MASH/fibrosis scoring. |
| `evaluate_translation.py` | Computes cycle consistency, orthologue preservation, and latent MMD for ARCHS4 folds. | `../archs4/evaluation/cycle_*`, `orthologue_*`, `mmd_*`; plotted by `../postprocessing/plot_archs4_evaluation.R`. |
| `evaluate_expression_mmd_archs4.py` | Computes expression-space MMD for ARCHS4 translated samples. | `../archs4/evaluation/expression_mmd_fold*.csv`; plotted by `plot_archs4_evaluation.R`. |
| `evaluate_liver.py` | Evaluates held-out liver reconstruction, cycle, orthologue, expression MMD, latent MMD, and centroid specificity metrics. | `../archs4/evaluation/liver_*_fold*.csv`; plotted by `plot_archs4_liver_evaluation.R`. |
| `score_liver_mas_fibrosis_final_expression_mean.py` | Averages full-ensemble translated expression and scores mouse MASH/fibrosis studies with human PLSR models. | `../archs4/evaluation/liver_mas_fibrosis_final_expression_mean/`; plotted by `plot_liver_mas_fibrosis_final_expression_mean.R`. |
| `AutoTransOP_Pretrain_FlowMatch.py` | Main unpaired FlowTransOP L1000 benchmark on shared-feature cell-line pairs. | `../results/AutoTransOP_CellPairs/`; summarized by `evaluate5folds.R`. |
| `AutoTransOP_Pretrain_FlowMatch_withPairs.py` | Paired-constrained FlowTransOP comparison for cell-line pairs. | `../results/AutoTransOP_CellPairs_withPairs/`; summarized by `evaluate5folds.R`. |
| `AutoTransOP_Pretrain_FlowMatch_withSTRUCTURE.py` | Compares a STRUCTURE-based initial alignment/pretraining variant against TRANSACT-based workflows. | `../results/AutoTransOP_withSTRUCTURE/`; summarized by `evaluate5folds.R`. |
| `DecodeFromConsencusSpace.py` | Decoder-only baseline using a consensus latent space for the shared-feature L1000 benchmark. | `../results/DecodersOnly/`; summarized by `evaluate5folds.R`. |
| `DecodeFromConsencusSpaceRandomPairs.py` | Decoder-only baseline with random/unpaired consensus construction. | Supports consensus-space control comparisons in the L1000 benchmark. |
| `AutoTransOP_Pretrain_FlowMatch_differentInputs.py` | FlowTransOP benchmark where the two domains use different feature subsets. | `../results/AutoTransOP_CellPairs_diffenetInputs/`; summarized by `evaluate5folds.R` and `DifferentInputsPerformanceBracketed.R`. |
| `DecodeFromConsencusSpace_diffenetInputs.py` | Decoder-only baseline for different-input L1000 tasks. | `../results/DecodersOnly_differentInputs/`; summarized by `evaluate5folds.R` and `DifferentInputsPerformanceBracketed.R`. |
| `AutoTransOP_Pretrain_FlowMatch_differentInputs_bracketed.py` | FlowTransOP across bracketed feature-correlation difficulty levels. | `../results/AutoTransOP_CellPairs_diffenetInputs_bracketed/`; summarized by `DifferentInputsPerformanceBracketed.R`. |
| `DecodeFromConsencusSpace_diffenetInputs_bracketed.py` | Decoder-only bracketed-difficulty baseline. | `../results/Decoders_only_diffenetInputs_bracketed/`; summarized by `DifferentInputsPerformanceBracketed.R`. |
| `AutoTransOP_Pretrain_FlowMatch_lowPairsPercentage.py` | FlowTransOP low-pair A375/HT29 benchmark. | `../results/FlowMatch_lowPairsPercentage/`; summarized by `LowPairsPerformance.R`. |
| `AutoTransOP_Pretrain_FlowMatchPaired_lowPairsPercentage.py` | Paired FlowTransOP low-pair A375/HT29 benchmark. | `../results/FlowMatch_lowPairsPercentage_withPairs/`; summarized by `LowPairsPerformance.R`. |
| `AutoTransOP_Pretrain_FlowMatch_lowPairsPercentageExtreme.py` | FlowTransOP extreme few-pair benchmark. | `../results/FlowMatch_extremely_fewPairs_A375_HT29/`; summarized by `LowPairsPerformance.R`. |
| `AutoTransOP_Pretrain_FlowMatch_lowPairsPercentageExtreme_withPairs.py` | Paired FlowTransOP extreme few-pair benchmark. | `../results/FlowMatch_extremely_fewPairs_A375_HT29_withPairs/`; summarized by `LowPairsPerformance.R`. |
| `AutoTransOP_lowPairsPercentageExtreme.py` | AutoTransOP baseline for the extreme few-pair benchmark. | `../results/AutoTransOP_extremely_fewPairs_A375_HT29/`; summarized by `LowPairsPerformance.R`. |
| `FlowMatch_lowPairsPercentage_PairsAndSimilarity.py` | Hybrid low-pair FlowTransOP variant using pair and similarity information. | `../results/FlowMatch_fewPairs_A375_HT29_PairAndSimilarity*/`; summarized by `LowPairsPerformance.R`. |
| `FlowMatch_lowPairsPercentageExtreme_PairsAndSimilarity.py` | Hybrid extreme few-pair variant with pair and similarity information. | `../results/FlowMatch_extremely_fewPairs_A375_HT29_PairAndSimilarity/`; summarized by `LowPairsPerformance.R`. |
| `FlowMatch_lowPairsPercentageExtreme_PairsAndSimilarity_meanAgg.py` | Mean-aggregation version of the pair-and-similarity variant. | `../results/*_PairAndSimilarity_meanAgg/`; summarized by `LowPairsPerformance.R`. |
| `FlowMatch_lowPairsPercentageExtreme_PairsAndSimilarity_sumAgg.py` | Sum-aggregation version of the pair-and-similarity variant. | `../results/*_PairAndSimilarity_sumAgg/`; summarized by `LowPairsPerformance.R`. |
| `InitialAligner_GPUvsCPU_random_data.py` | TRANSACT GPU/CPU comparison on synthetic/random data. | `../results/GPU_vs_CPU_random/`; plotted by `GPU_vs_CPU_implementation.R`. |
| `InitialAligner_GPUvsCPU_celline_pairs.py` | TRANSACT GPU/CPU comparison on real cell-line pair data. | `../results/GPU_vs_CPU/`; plotted by `GPU_vs_CPU_implementation.R`. |
| `InitialAligner_GPUvsCPU_celline_pairs_random_subsampling.py` | GPU/CPU TRANSACT comparison under real-data subsampling. | Supports `GPU_vs_CPU_implementation.R` comparisons. |
| `InitialAligner_GPUvsCPU_sameCell_diffInput.py` | GPU/CPU TRANSACT comparison for same-cell, different-feature inputs. | Supports `GPU_vs_CPU_implementation.R` comparisons. |
| `models.py` | FlowTransOP neural-network modules used by the main training scripts. | Imported by training, inference, and package loading code. |
| `models_autotransop.py` | Neural modules for AutoTransOP-style baselines. | Imported by AutoTransOP baseline scripts. |
| `trainingUtils.py` | Training loops, losses, and helpers for FlowTransOP scripts. | Imported by FlowTransOP training scripts. |
| `trainingUtils_autotransop.py` | Training helpers for AutoTransOP-style baselines. | Imported by AutoTransOP baseline scripts. |
| `transact_utility_gpu.py` | GPU-oriented TRANSACT/pre-alignment utilities. | Imported by GPU/pretraining workflows. |
| `utility.py` | General utility functions, including CPU-side alignment/helper routines used by benchmarks. | Imported across L1000 scripts and GPU/CPU comparison scripts. |
| `evaluationUtils.py` | Metric and statistics helpers for benchmark evaluation. | Imported by L1000 and ARCHS4 evaluation scripts. |

## Shell Wrapper Map

| Wrapper | What it runs |
| --- | --- |
| `retrieve_ARCHS4.sh` | SLURM wrapper for `archs4_workflow.py`. |
| `preprocess_ARCHS4.sh` | SLURM wrapper for `preprocess_archs4.py`. |
| `mouse_preprocess.sh` | SLURM wrapper for `preprocess_archs4_mouse.py`. |
| `mouse_preprocess_preemptable.sh` | Preemptable SLURM wrapper for `preprocess_archs4_mouse.py`. |
| `ARCHS4_train_CV.sh` | SLURM array wrapper for `train_ARCHS4_fold.py`. |
| `ARCHS4_train_full_ensemble.sh` | SLURM array wrapper for `train_ARCHS4_full_ensemble.py`. |
| `evaluate_translation.sh` | SLURM array wrapper for `evaluate_translation.py`. |
| `score_liver_mas_fibrosis_final_expression_mean.sh` | SLURM wrapper for final MASH/fibrosis scoring. |
| `cell_pairs_benchmark.sh` | Historical SLURM wrapper for the shared-feature L1000 benchmark; edit the active Python command as needed. |
| `low_percentage_of_pairs.sh` | Runs `AutoTransOP_Pretrain_FlowMatch_lowPairsPercentage.py`. |
| `extremely_low_percentage_of_pairs.sh` | Runs `AutoTransOP_Pretrain_FlowMatch_lowPairsPercentageExtreme.py`. |
| `pairedFlow_low_percentage_of_pairs.sh` | Runs `AutoTransOP_Pretrain_FlowMatchPaired_lowPairsPercentage.py`. |
| `pairedFlow_low_percentage_of_pairs_extreme.sh` | Runs `AutoTransOP_Pretrain_FlowMatch_lowPairsPercentageExtreme_withPairs.py`. |
| `OneCell_differentInputs_benchmark.sh` | Runs `AutoTransOP_Pretrain_FlowMatch_differentInputs_bracketed.py`. |
| `decoders_only_imputedGenes.sh` | Runs `DecodeFromConsencusSpace_diffenetInputs_bracketed.py`. |
| `subsetting_decoders_only.sh` | Runs `DecodeFromConsencusSpaceRandomPairs.py` for decoder-only random-pair/subsetting controls. |
| `run_liver_translated_expression_pathway_activity.sh` | Historical wrapper for a Hallmark ssGSEA analysis script; restore the companion Python script if this optional analysis is needed. |

## Package Entry Point

The repository root contains an installable package. From `..`:

```bash
pip install -e ".[reproduce]"
```

Then from the repository root:

```bash
flowtransop train-archs4-fold --repo-root . --fold 0 --direction h2m \
  --model-device cuda --transact-backend gpu --transact-device cuda
flowtransop train-archs4-fold --repo-root . --fold 0 --direction m2h \
  --model-device cuda --transact-backend gpu --transact-device cuda
flowtransop evaluate-archs4-fold --repo-root . --fold 0 --include-liver
```

`--model-device` controls the model device. `--transact-backend` and
`--transact-device` configure TRANSACT/pre-alignment separately from the model
device. Defaults are GPU/CUDA:

```text
--model-device cuda
--transact-backend gpu
--transact-device cuda
```

Python users can make the same choice directly:

```python
from flowtransop import RuntimeBackends, load_transact_backend

backends = RuntimeBackends(model_device="cuda", transact_backend="cpu", transact_device="cpu")
transact = load_transact_backend(repo_root=".", backends=backends)
Z_source, Z_target, tau, model = transact.align(X_source, X_target)
```

For direct inference on preprocessed matrices:

```bash
flowtransop predict \
  --normal-checkpoint archs4/models/fold_0_normal.pt \
  --m2h-checkpoint archs4/models/fold_0_normal_m2h.pt \
  --direction m2h \
  --input-npy archs4/preprocessed/mouse_test_X.npy \
  --output-npy archs4/evaluation/example_m2h_prediction.npy
```

## Practical Notes

- ARCHS4 training is large-scale and expects a GPU with substantial memory.
- The shell scripts are configured for the MIT/Lauffenburger SLURM environment;
  edit only scheduler headers and environment activation commands when porting.
- All scripts assume feature order is fixed. For inference, inputs must match
  the corresponding `*_genes.npy` order and preprocessing used during training.
- Keep permuted-feature controls when reproducing results; they define the
  random baseline for unpaired translation.
