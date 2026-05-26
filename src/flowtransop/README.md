# FlowTransOP package scaffold

This folder contains the lightweight installable package for the repository.
It does not replace the research scripts in `learning/`; instead, it wraps the
existing checkpoint format and common entry points so users can install a
`flowtransop` command.

## Install

From the repository root:

```bash
pip install -e .
```

For the full manuscript workflow, install the optional reproduction extras:

```bash
pip install -e ".[reproduce]"
```

## Predict with a trained ARCHS4 checkpoint

Inputs must already be preprocessed and ordered exactly like the model training
matrix. For the ARCHS4 models this means human inputs for `h2m` must follow
`archs4/preprocessed/human_genes.npy`, and mouse inputs for `m2h` must follow
`archs4/preprocessed/mouse_genes.npy`.

```bash
flowtransop predict \
  --normal-checkpoint archs4/models/full_ensemble_0_normal.pt \
  --direction h2m \
  --input-npy path/to/preprocessed_human_samples.npy \
  --output-npy translated_mouse_expression.npy
```

For CV checkpoints, mouse-to-human translation usually stores the reverse flow
in a separate file:

```bash
flowtransop predict \
  --normal-checkpoint archs4/models/fold_0_normal.pt \
  --m2h-checkpoint archs4/models/fold_0_normal_m2h.pt \
  --direction m2h \
  --input-npy path/to/preprocessed_mouse_samples.npy \
  --output-npy translated_human_expression.npy
```

## Run repository workflows through the CLI

The CLI can call the original scripts from `learning/`:

```bash
flowtransop train-archs4-fold --repo-root . --fold 0 --direction h2m
flowtransop train-archs4-fold --repo-root . --fold 0 --direction m2h
flowtransop train-archs4-ensemble --repo-root . --ensemble-id 0 --fold 0
flowtransop evaluate-archs4-fold --repo-root . --fold 0 --include-liver
flowtransop score-mash --repo-root .
```

Pass extra script arguments after the subcommand arguments, for example:

```bash
flowtransop train-archs4-ensemble --repo-root . --ensemble-id 0 --fold 0 --epochs 10 --batch_size 2048
```

The original scripts remain the most complete source for all experimental
options. This package is intentionally small so users can load models and run
standard workflows without reorganizing the repository.
