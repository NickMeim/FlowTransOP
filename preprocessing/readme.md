# Preprocessing

This folder contains preprocessing scripts for the L1000 benchmark portion of
the FlowTransOP study.

ARCHS4 preprocessing is handled by the Python scripts in `../learning/`
(`archs4_workflow.py`, `preprocess_archs4.py`, and
`preprocess_archs4_mouse.py`) because those scripts write directly to
`../archs4/`.

## L1000 Preprocessing

The main script is:

```bash
Rscript preProcessL1000DrugData.R
```

Run it from this folder:

```bash
cd preprocessing
Rscript preProcessL1000DrugData.R
```

The expected output location is:

```text
preprocessing/preprocessed_data/
```

The downstream scripts in `../learning/` expect processed L1000 matrices,
metadata, cell-line pair definitions, and train/test fold information to be
available from this preprocessing output or from the AutoTransOP-compatible
processed files used in the manuscript.

## Expected Downstream Use

After preprocessing, run model training from `../learning/`, for example:

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
