# ARCHS4 Splits

This folder stores split definitions and metadata for the ARCHS4 mouse-human
experiments. The JSON split files and metadata tables can be large, so they are
kept out of GitHub by default.

Expected local files include:

- `human_split.json`
- `mouse_split.json`
- `human_folds.json`
- `mouse_folds.json`
- `liver_metadata_human.csv`
- `liver_metadata_mouse.csv`

These files are created or consumed by the ARCHS4 preprocessing workflow in
`learning/`, including `archs4_workflow.py` and the associated SLURM wrappers.
