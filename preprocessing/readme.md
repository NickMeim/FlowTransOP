# Preprocessing

This folder contains preprocessing material for the L1000 benchmark portion of
the FlowTransOP study.

ARCHS4 preprocessing is handled by the Python scripts in `../learning/`
(`archs4_workflow.py`, `preprocess_archs4.py`, and
`preprocess_archs4_mouse.py`) because those scripts write directly to
`../archs4/`.

## Active L1000 Inputs Used by the Manuscript Scripts

The L1000 learning scripts used for the manuscript expect AutoTransOP-compatible
processed files under:

```text
preprocessing/preprocessed_data/CellPairs/
preprocessing/preprocessed_data/SameCellimputationModel/
preprocessing/preprocessed_data/SameCellimputationModel/bracketed_difficulty/
```

Common files referenced by downstream scripts include:

```text
preprocessing/preprocessed_data/CellPairs/drug_landmarks.csv
preprocessing/preprocessed_data/CellPairs/cmap_all_genes_q1_tas03.csv
preprocessing/preprocessed_data/CellPairs/train_paired_{fold}.csv
preprocessing/preprocessed_data/CellPairs/val_paired_{fold}.csv
```

The learning scripts in `../learning/` read these processed matrices, cell-line
pair definitions, and fold splits, then write benchmark outputs to
`../results/`.

## Legacy Script: `preProcessL1000DrugData.R`

`preProcessL1000DrugData.R` appears to be a legacy or exploratory preprocessing
script. It creates pair, triplet, and quadruplet split objects such as:

```text
preprocessed_data/drug_pairs_dataset.rds
preprocessed_data/drug_triplets_dataset.rds
preprocessed_data/drug_quadruplets_dataset.rds
preprocessed_data/drug_test_sets_pairs.rds
preprocessed_data/drug_tuning_and_cv_pairs.json
preprocessed_data/drug_tuning_and_cv_triplets.json
preprocessed_data/drug_tuning_and_cv_quadruplets_part*.json
```

A repository-wide search did not find these exact outputs being consumed by the
current learning or postprocessing scripts. In other words, this script is not
part of the active manuscript reproduction path unless you intentionally want to
recreate that older pair/triplet/quadruplet split format.

If you do run it, execute it from this folder:

```bash
cd preprocessing
Rscript preProcessL1000DrugData.R
```

## Downstream Workflow

After the active preprocessed L1000 files are available, run model training from
`../learning/`, for example:

```bash
cd ../learning
bash cell_pairs_benchmark.sh
bash low_percentage_of_pairs.sh
bash OneCell_differentInputs_benchmark.sh
```

Outputs are written under `../results/` and plotted by scripts in
`../postprocessing/`.

## Notes

- Keep feature ordering and sample metadata synchronized with the training
  scripts. The benchmark scripts assume the same L1000 splits and cell-line
  tasks used for the manuscript.
- Large intermediate files should remain under `preprocessed_data/` rather than
  being moved into script folders.
- If reproducing only the ARCHS4 case study, this folder is not required; use
  the ARCHS4 workflow described in `../README.md` and `../learning/README.md`.
