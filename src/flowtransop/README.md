# FlowTransOP package scaffold

This folder contains the lightweight installable package for the repository.
It does not replace the research scripts in `learning/`; instead, it wraps the
existing checkpoint format and common entry points so users can install a
`flowtransop` command.

## Install

Use Python 3.9 or newer. From the repository root, a minimal install for package
import, CLI help, and checkpoint inference is:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

For the full manuscript workflow and any CLI command that delegates to the
original training/evaluation scripts in `learning/`, install the optional
reproduction extras. These include script-level dependencies such as
`statsmodels`, `h5py`, `archs4py`, `geomloss`, and plotting packages:

```bash
python -m pip install -e ".[reproduce]"
```

On SLURM, the repository-level smoke test validates the installable package in a
fresh virtual environment:

```bash
sbatch archived/flowtransop_package_smoke_test.slurm.sh
```

The smoke test requires Python >= 3.9, refuses to continue if the virtual
environment fails to activate, installs `.[reproduce]` when the L1000 smoke
training is enabled, and writes outputs under
`archived/flowtransop_smoke_${SLURM_JOB_ID}/`.

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

It can also launch the L1000 approaches discussed in the manuscript:

These options are part of the default installed `flowtransop` command:

| Method argument | Workflow |
| --- | --- |
| `--method hybrid-flowtransop` | Hybrid FlowTransOP using pair and pre-aligned similarity constraints. |
| `--method simple-autotransop` or `--method autotransop` | AutoTransOP/CPA-style baseline. The CLI prints a hyperparameter-sensitivity warning when this is run. |
| `--method consensus-decoders` | Consensus-space decoder baseline. |

```bash
flowtransop run-l1000 --repo-root . --method consensus-decoders
flowtransop run-l1000 --repo-root . --method hybrid-flowtransop
flowtransop run-l1000 --repo-root . --method simple-autotransop
```

Available L1000 methods are:

```text
flowtransop
consensus-decoders
consensus-decoders-different-inputs
consensus-decoders-bracketed
hybrid-flowtransop
hybrid-flowtransop-extreme
hybrid-flowtransop-extreme-mean
hybrid-flowtransop-extreme-sum
autotransop
simple-autotransop
```

**Important AutoTransOP note:** AutoTransOP hyperparameters are very important
and the method can be highly sensitive. Whether to use mutual information,
cosine distance, Euclidean distance, and/or prior/adversarial discriminators as
proposed in the original publication is a modeling choice that users must
customly re-adjust for their own data, paired-sample regime, and feature space.
The CLI prints this note every time `--method autotransop` or
`--method simple-autotransop` is run.

## Use the pretrained ARCHS4 ensemble

The 10 full ARCHS4 ensemble checkpoints are too large for GitHub and are
expected to live in the local, gitignored `archs4/` folder:

```text
archs4/models/full_ensemble_0_normal.pt
...
archs4/models/full_ensemble_9_normal.pt
```

Average predictions across the 10 pretrained ensemble members:

```bash
flowtransop predict-archs4-ensemble \
  --archs4-dir archs4 \
  --ensemble-ids 0-9 \
  --direction h2m \
  --input-npy archs4/preprocessed/human_test_X.npy \
  --output-npy archs4/evaluation/example_ensemble_h2m_prediction.npy
```

Optionally save each member prediction as a separate array:

```bash
flowtransop predict-archs4-ensemble \
  --archs4-dir archs4 \
  --ensemble-ids 0-9 \
  --direction m2h \
  --input-npy archs4/preprocessed/mouse_test_X.npy \
  --output-npy archs4/evaluation/example_ensemble_m2h_prediction.npy \
  --members-output-npy archs4/evaluation/example_ensemble_m2h_members.npy
```

Python API:

```python
from flowtransop import load_archs4_ensemble

ensemble = load_archs4_ensemble(archs4_dir="archs4", ensemble_ids="0-9")
translated = ensemble.translate(mouse_expression, direction="m2h")
```

Fine-tune one pretrained ensemble member when the full preprocessed ARCHS4 data
are available locally:

```bash
flowtransop finetune-archs4-ensemble \
  --repo-root . \
  --archs4-dir archs4 \
  --ensemble-id 0 \
  --epochs 5 \
  --output-model-dir archs4/models_finetuned
```

Fine-tuning starts from `archs4/models/full_ensemble_{id}_normal.pt` and writes
new checkpoints under `archs4/models_finetuned` by default.

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
python -m pip install -e ".[reproduce]"

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
