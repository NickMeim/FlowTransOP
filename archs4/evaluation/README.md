# ARCHS4 Evaluation

This folder contains evaluation CSVs and figure outputs for the ARCHS4
experiments. Unlike the raw ARCHS4 data and model checkpoints, these outputs are
intended to be tracked when they are below 90 MB.

Main generators in `learning/`:

- `evaluate_translation.py`: cyclic reconstruction and orthologue evaluation.
- `evaluate_expression_mmd_archs4.py`: expression-space MMD summaries.
- `evaluate_liver.py`: held-out liver evaluation metrics.
- `score_liver_mas_fibrosis_final_expression_mean.py`: MASH and fibrosis case
  study scoring.

Main plotting scripts in `postprocessing/`:

- `plot_archs4_evaluation.R`
- `plot_archs4_liver_evaluation.R`
- `plot_liver_mas_fibrosis_final_expression_mean.R`

Git cannot filter by file size, so any future evaluation artifact above 90 MB
should stay untracked or be moved to external storage.
