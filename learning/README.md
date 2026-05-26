# FlowTransOP Learning Scripts

This folder contains the Python and shell scripts used for model training,
evaluation, and disease-score analysis. Most paths in these scripts are relative
to this folder, so run commands from `learning/` unless noted otherwise.

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
python archs4_workflow.py
python preprocess_archs4.py
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
python train_ARCHS4_fold.py --fold 0
python train_ARCHS4_fold_m2h.py --fold 0
```

The first command trains encoders, decoders, human-to-mouse flow, and permuted
controls. The second command loads those saved encoders/decoders and trains the
mouse-to-human flow.

SLURM wrappers:

```bash
sbatch --array=0-9 ARCHS4_train_CV.sh
```

Train full-data ensemble models for the MASH case study:

```bash
python train_ARCHS4_full_ensemble.py --fold 0 --ensemble_id 0
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
python evaluate_translation.py --fold 0
python evaluate_expression_mmd_archs4.py --fold 0
python evaluate_liver.py --fold 0
```

`evaluate_translation.py` produces cycle consistency, orthologue preservation,
and latent MMD metrics. `evaluate_expression_mmd_archs4.py` adds expression
space MMD. `evaluate_liver.py` evaluates held-out liver reconstruction, cycle
consistency, orthologues, MMD, and centroid metrics.

### MASH/Fibrosis Scoring

After training full ensemble models:

```bash
python score_liver_mas_fibrosis_final_expression_mean.py --ensemble_ids 0-9
```

The script averages translated expression across ensemble members, trains human
PLSR models for MAS/NAS score and fibrosis stage, writes LOOCV performance, and
scores the selected mouse treatment studies.

Outputs:

```text
../archs4/evaluation/liver_mas_fibrosis_final_expression_mean/
```

## Package Entry Point

The repository root contains an installable package. From `..`:

```bash
pip install -e .
```

Then from the repository root:

```bash
flowtransop train-archs4-fold --repo-root . --fold 0 --direction h2m
flowtransop train-archs4-fold --repo-root . --fold 0 --direction m2h
flowtransop evaluate-archs4-fold --repo-root . --fold 0 --include-liver
```

For direct inference on preprocessed matrices:

```bash
flowtransop predict \
  --normal-checkpoint archs4/models/full_ensemble_0_normal.pt \
  --direction m2h \
  --input-npy path/to/preprocessed_mouse_samples.npy \
  --output-npy translated_human_expression.npy
```

## Practical Notes

- ARCHS4 training is large-scale and expects a GPU with substantial memory.
- The shell scripts are configured for the MIT/Lauffenburger SLURM environment;
  edit only scheduler headers and environment activation commands when porting.
- All scripts assume feature order is fixed. For inference, inputs must match
  the corresponding `*_genes.npy` order and preprocessing used during training.
- Keep permuted-feature controls when reproducing results; they define the
  random baseline for unpaired translation.
