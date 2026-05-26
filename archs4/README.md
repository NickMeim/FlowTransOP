# ARCHS4 Workspace

This directory is the local workspace for the ARCHS4 mouse-human experiments.
It is intentionally mostly gitignored because the raw HDF5 files, processed
matrices, fold splits, latent arrays, and model checkpoints are too large for
GitHub.

Tracked contents should stay lightweight:

- `README.md` files that document the expected folder structure.
- Evaluation outputs under `evaluation/` that are below the repository size
  limit.

Expected local subfolders:

- `splits/`: ARCHS4 train/validation fold definitions and liver metadata.
- `models/`: cross-validation and full-ensemble FlowTransOP checkpoints plus
  latent arrays.
- `evaluation/`: CSV summaries and figure outputs used by the manuscript.

The study scripts usually refer to this directory as `../archs4` when run from
`learning/`.
