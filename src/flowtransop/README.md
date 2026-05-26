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
flowtransop train-archs4-fold --repo-root . --fold 0 --direction h2m \
  --model-device cuda --transact-backend gpu --transact-device cuda
flowtransop train-archs4-fold --repo-root . --fold 0 --direction m2h \
  --model-device cuda --transact-backend gpu --transact-device cuda
flowtransop train-archs4-ensemble --repo-root . --ensemble-id 0 --fold 0 \
  --model-device cuda --transact-backend gpu --transact-device cuda
flowtransop evaluate-archs4-fold --repo-root . --fold 0 --include-liver
flowtransop score-mash --repo-root .
```

`--model-device` controls the Torch model device. `--transact-backend` and
`--transact-device` are separate so TRANSACT/pre-alignment can be configured
independently of model training. The defaults are GPU/CUDA:

```text
--model-device cuda
--transact-backend gpu
--transact-device cuda
```

For CPU pre-alignment with GPU model training:

```bash
flowtransop train-archs4-fold --repo-root . --fold 0 --direction h2m \
  --model-device cuda --transact-backend cpu --transact-device cpu
```

The same separation is available from Python:

```python
from flowtransop import RuntimeBackends, load_transact_backend

backends = RuntimeBackends(
    model_device="cuda",
    transact_backend="cpu",
    transact_device="cpu",
)
transact = load_transact_backend(repo_root=".", backends=backends)
Z_source, Z_target, tau, model = transact.align(X_source, X_target)
```

Pass extra script arguments after the subcommand arguments, for example:

```bash
flowtransop train-archs4-ensemble --repo-root . --ensemble-id 0 --fold 0 \
  --model-device cuda --transact-backend gpu --transact-device cuda \
  --epochs 10 --batch_size 2048
```

The original scripts remain the most complete source for all experimental
options. This package is intentionally small so users can load models and run
standard workflows without reorganizing the repository.

## Minimal train, evaluate, predict example

From the repository root, after ARCHS4 preprocessing has created
`archs4/preprocessed/`:

```bash
pip install -e ".[reproduce]"

flowtransop train-archs4-fold --repo-root . --fold 0 --direction h2m
flowtransop train-archs4-fold --repo-root . --fold 0 --direction m2h
flowtransop evaluate-archs4-fold --repo-root . --fold 0 --include-liver

flowtransop predict \
  --normal-checkpoint archs4/models/fold_0_normal.pt \
  --m2h-checkpoint archs4/models/fold_0_normal_m2h.pt \
  --direction m2h \
  --input-npy archs4/preprocessed/mouse_test_X.npy \
  --output-npy archs4/evaluation/example_m2h_prediction.npy
```
