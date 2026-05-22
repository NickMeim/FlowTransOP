#!/usr/bin/env python3
"""
UMAP visualization of full-ensemble ARCHS4 mouse-human translation.

This script uses the full-data ARCHS4 ensemble checkpoints to build two
target-space visualizations:

  1. Human samples translated into mouse expression space, plotted with mouse
     target-reference samples.
  2. Mouse samples translated into human expression space, plotted with human
     target-reference samples.

The default is intentionally conservative for decoder-generated expression:
real target samples are also reconstructed through the target autoencoders and
averaged across the ensemble before UMAP. This avoids a trivial UMAP separation
between raw high-variance target expression and smoother decoder means. Pass
--target_reference raw if you explicitly want raw target expression instead.

For each direction, the script:
  * uses all source/target samples by default, or a representative subset if requested,
  * computes ensemble-mean translated expression for source samples,
  * computes ensemble-mean target reconstruction or uses raw target expression,
  * selects target-space HVGs from both raw target and translated source data,
  * mean-centers genes, PCA-denoises, and runs UMAP,
  * writes coordinates, sample metadata, PCA variance, selected genes, and PNGs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from models import ElementWiseLinear, Flow, SimpleEncoder, VarDecoder


DATA_DIR = Path("../archs4")
PREPROC_DIR = DATA_DIR / "preprocessed"
MODEL_DIR = DATA_DIR / "models"
OUT_DIR = DATA_DIR / "evaluation" / "archs4_full_ensemble_umap"


def require_file(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return path


def parse_int_ranges(text: str) -> list[int]:
    values: list[int] = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start, stop = piece.split("-", 1)
            values.extend(range(int(start), int(stop) + 1))
        else:
            values.append(int(piece))
    values = sorted(set(values))
    if not values:
        raise ValueError("No ensemble ids were parsed.")
    return values


def activation(name: str):
    return {
        "LeakyReLU": torch.nn.LeakyReLU(0.01),
        "ReLU": torch.nn.ReLU(),
        "ELU": torch.nn.ELU(),
        "Sigmoid": torch.nn.Sigmoid(),
    }[name]


def model_params_from_checkpoint_args(raw_args: dict) -> dict:
    return {
        "encoder_1_hiddens": raw_args["encoder_1_hiddens"],
        "encoder_2_hiddens": raw_args["encoder_2_hiddens"],
        "latent_dim": raw_args["latent_dim"],
        "decoder_1_hiddens": raw_args["decoder_1_hiddens"],
        "decoder_2_hiddens": raw_args["decoder_2_hiddens"],
        "dropout_decoder": raw_args["dropout_decoder"],
        "dropout_encoder": raw_args["dropout_encoder"],
        "encoder_activation": activation(raw_args["encoder_activation"]),
        "decoder_activation": activation(raw_args["decoder_activation"]),
        "bn_encoder": raw_args["bn_encoder"],
        "bn_decoder": raw_args["bn_decoder"],
        "dropout_input_encoder": raw_args["dropout_input_encoder"],
        "dropout_input_decoder": raw_args["dropout_input_decoder"],
    }


def build_models(model_params: dict, human_dim: int, mouse_dim: int, device: torch.device):
    enc_h = torch.nn.Sequential(
        ElementWiseLinear(human_dim),
        SimpleEncoder(
            human_dim,
            model_params["encoder_1_hiddens"],
            model_params["latent_dim"],
            dropRate=model_params["dropout_encoder"],
            bn=model_params["bn_encoder"],
            activation=model_params["encoder_activation"],
            dropIn=model_params["dropout_input_encoder"],
            dtype=torch.float,
        ),
    ).to(device)
    enc_m = torch.nn.Sequential(
        ElementWiseLinear(mouse_dim),
        SimpleEncoder(
            mouse_dim,
            model_params["encoder_2_hiddens"],
            model_params["latent_dim"],
            dropRate=model_params["dropout_encoder"],
            bn=model_params["bn_encoder"],
            activation=model_params["encoder_activation"],
            dropIn=model_params["dropout_input_encoder"],
            dtype=torch.float,
        ),
    ).to(device)
    dec_h = VarDecoder(
        model_params["latent_dim"],
        model_params["decoder_1_hiddens"],
        human_dim,
        dropRate=model_params["dropout_decoder"],
        bn=model_params["bn_decoder"],
        activation=model_params["decoder_activation"],
        dropIn=model_params["dropout_input_decoder"],
        loss="gauss",
        dtype=torch.float,
    ).to(device)
    dec_m = VarDecoder(
        model_params["latent_dim"],
        model_params["decoder_2_hiddens"],
        mouse_dim,
        dropRate=model_params["dropout_decoder"],
        bn=model_params["bn_decoder"],
        activation=model_params["decoder_activation"],
        dropIn=model_params["dropout_input_decoder"],
        loss="gauss",
        dtype=torch.float,
    ).to(device)
    flow_h2m = Flow(
        model_params["latent_dim"],
        model_params["latent_dim"] // 2,
        dtype=torch.float,
    ).to(device)
    flow_m2h = Flow(
        model_params["latent_dim"],
        model_params["latent_dim"] // 2,
        dtype=torch.float,
    ).to(device)
    return enc_h, enc_m, dec_h, dec_m, flow_h2m, flow_m2h


def load_ensemble_member(
    args: argparse.Namespace,
    ensemble_id: int,
    human_dim: int,
    mouse_dim: int,
    device: torch.device,
) -> dict:
    normal_path = require_file(
        args.model_dir / f"{args.full_model_prefix}_{ensemble_id}_normal.pt"
    )
    ckpt = torch.load(normal_path, map_location=device, weights_only=False)
    raw_args = ckpt["args"]
    params = model_params_from_checkpoint_args(raw_args)
    enc_h, enc_m, dec_h, dec_m, flow_h2m, flow_m2h = build_models(
        params,
        human_dim,
        mouse_dim,
        device,
    )
    enc_h.load_state_dict(ckpt["encoder_human"])
    enc_m.load_state_dict(ckpt["encoder_mouse"])
    dec_h.load_state_dict(ckpt["decoder_human"])
    dec_m.load_state_dict(ckpt["decoder_mouse"])
    flow_h2m.load_state_dict(ckpt["flow_h2m"])
    if "flow_m2h" in ckpt:
        flow_m2h.load_state_dict(ckpt["flow_m2h"])
    else:
        m2h_path = require_file(
            args.model_dir / f"{args.full_model_prefix}_{ensemble_id}_normal_m2h.pt"
        )
        ckpt_m2h = torch.load(m2h_path, map_location=device, weights_only=False)
        flow_m2h.load_state_dict(ckpt_m2h["flow_m2h"])

    modules = [enc_h, enc_m, dec_h, dec_m, flow_h2m, flow_m2h]
    for module in modules:
        module.eval()
    return {
        "enc_h": enc_h,
        "enc_m": enc_m,
        "dec_h": dec_h,
        "dec_m": dec_m,
        "flow_h2m": flow_h2m,
        "flow_m2h": flow_m2h,
    }


@dataclass
class MatrixPart:
    label: str
    matrix: np.ndarray
    sample_ids: np.ndarray


@dataclass
class SelectedDataset:
    species: str
    parts: list[MatrixPart]
    part_indices: np.ndarray
    row_indices: np.ndarray
    sample_ids: np.ndarray
    split_labels: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(len(self.sample_ids))

    @property
    def n_features(self) -> int:
        return int(self.parts[0].matrix.shape[1])

    def iter_batches(self, batch_size: int, cols: np.ndarray | None = None):
        n_features = self.n_features if cols is None else int(len(cols))
        for start in range(0, self.n_samples, batch_size):
            stop = min(start + batch_size, self.n_samples)
            part_slice = self.part_indices[start:stop]
            row_slice = self.row_indices[start:stop]
            block = np.empty((stop - start, n_features), dtype=np.float32)
            for part_i in np.unique(part_slice):
                mask = part_slice == part_i
                rows = row_slice[mask]
                part_block = np.asarray(self.parts[int(part_i)].matrix[rows, :], dtype=np.float32)
                if cols is not None:
                    part_block = part_block[:, cols]
                block[mask] = part_block
            yield start, stop, block

    def to_numpy(self, batch_size: int, cols: np.ndarray | None = None) -> np.ndarray:
        n_features = self.n_features if cols is None else int(len(cols))
        out = np.empty((self.n_samples, n_features), dtype=np.float32)
        for start, stop, block in self.iter_batches(batch_size, cols=cols):
            out[start:stop] = block
        return out

    def metadata(self, point_type: str, target_space: str, ensemble_sd: np.ndarray | None = None):
        frame = pd.DataFrame(
            {
                "sample_id": self.sample_ids.astype(str),
                "original_species": self.species,
                "target_space": target_space,
                "point_type": point_type,
                "split": self.split_labels.astype(str),
            }
        )
        if ensemble_sd is not None:
            frame["mean_genewise_ensemble_sd"] = ensemble_sd
        else:
            frame["mean_genewise_ensemble_sd"] = np.nan
        return frame


def load_species_parts(preproc_dir: Path, species: str, include_test: bool) -> list[MatrixPart]:
    parts = [
        MatrixPart(
            "train_pool",
            np.load(require_file(preproc_dir / f"{species}_X.npy"), mmap_mode="r"),
            np.load(require_file(preproc_dir / f"{species}_sample_ids.npy"), allow_pickle=True),
        )
    ]
    test_x = preproc_dir / f"{species}_test_X.npy"
    test_ids = preproc_dir / f"{species}_test_sample_ids.npy"
    if include_test and test_x.exists() and test_ids.exists():
        parts.append(
            MatrixPart(
                "liver_test",
                np.load(test_x, mmap_mode="r"),
                np.load(test_ids, allow_pickle=True),
            )
        )
    return parts


def choose_rows(
    parts: list[MatrixPart],
    n_samples: int | str,
    test_fraction: float,
    max_test_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    use_all = str(n_samples).lower() == "all"
    if not use_all and int(n_samples) <= 0:
        raise ValueError("n_samples must be positive or 'all'.")
    n_parts = len(parts)
    train_n = parts[0].matrix.shape[0]
    test_n = parts[1].matrix.shape[0] if n_parts > 1 else 0

    if use_all:
        train_rows = np.arange(train_n, dtype=np.int64)
        test_rows = np.arange(test_n, dtype=np.int64) if test_n > 0 else np.array([], dtype=np.int64)
    else:
        n_total = int(n_samples)
        n_test = 0
        if test_n > 0:
            n_test = min(test_n, max_test_samples, int(round(n_total * test_fraction)))
        n_train = min(train_n, max(0, n_total - n_test))
        if n_train + n_test < n_total and test_n > n_test:
            n_extra = min(test_n - n_test, n_total - n_train - n_test)
            n_test += n_extra

        train_rows = rng.choice(train_n, size=n_train, replace=False) if n_train > 0 else np.array([], dtype=int)
        test_rows = rng.choice(test_n, size=n_test, replace=False) if n_test > 0 else np.array([], dtype=int)

    n_train = len(train_rows)
    n_test = len(test_rows)

    part_indices = [np.zeros(n_train, dtype=np.int64)]
    row_indices = [train_rows.astype(np.int64)]
    sample_ids = [parts[0].sample_ids[train_rows].astype(str)]
    split_labels = [np.repeat(parts[0].label, n_train)]
    if n_test > 0:
        part_indices.append(np.ones(n_test, dtype=np.int64))
        row_indices.append(test_rows.astype(np.int64))
        sample_ids.append(parts[1].sample_ids[test_rows].astype(str))
        split_labels.append(np.repeat(parts[1].label, n_test))

    return (
        np.concatenate(part_indices),
        np.concatenate(row_indices),
        np.concatenate(sample_ids),
        np.concatenate(split_labels),
    )


def sample_species_dataset(
    preproc_dir: Path,
    species: str,
    n_samples: int | str,
    include_test: bool,
    test_fraction: float,
    max_test_samples: int,
    seed: int,
) -> SelectedDataset:
    parts = load_species_parts(preproc_dir, species, include_test)
    rng = np.random.default_rng(seed)
    part_indices, row_indices, sample_ids, split_labels = choose_rows(
        parts,
        n_samples,
        test_fraction,
        max_test_samples,
        rng,
    )
    order = rng.permutation(len(sample_ids))
    return SelectedDataset(
        species=species,
        parts=parts,
        part_indices=part_indices[order],
        row_indices=row_indices[order],
        sample_ids=sample_ids[order],
        split_labels=split_labels[order],
    )


def flow_step_n(flow, z: torch.Tensor, n_steps: int) -> torch.Tensor:
    time_steps = torch.linspace(0, 1.0, n_steps + 1, device=z.device, dtype=torch.float)
    out = z
    for step in range(n_steps):
        out = flow.step(out, time_steps[step], time_steps[step + 1])
    return out


def select_raw_target_variable_genes(
    target: SelectedDataset,
    target_genes: np.ndarray,
    top_n: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if top_n <= 0 or top_n >= target.n_features:
        idx = np.arange(target.n_features, dtype=np.int64)
        return idx, target_genes[idx]

    sums = np.zeros(target.n_features, dtype=np.float64)
    sums_sq = np.zeros(target.n_features, dtype=np.float64)
    n_total = 0
    for _, _, block in target.iter_batches(batch_size):
        block64 = block.astype(np.float64, copy=False)
        sums += block64.sum(axis=0)
        sums_sq += np.square(block64).sum(axis=0)
        n_total += block.shape[0]
    means = sums / float(n_total)
    variances = np.maximum(sums_sq / float(n_total) - means**2, 0.0)
    idx = np.argpartition(variances, -top_n)[-top_n:]
    idx = idx[np.argsort(variances[idx])[::-1]].astype(np.int64)
    return idx, target_genes[idx]


@torch.no_grad()
def member_translated_gene_variance(
    selected: SelectedDataset,
    member: dict,
    direction: str,
    device: torch.device,
    batch_size: int,
    flow_steps: int,
    out_dim: int,
) -> np.ndarray:
    if direction == "h2m":
        encoder, flow, decoder = member["enc_h"], member["flow_h2m"], member["dec_m"]
    elif direction == "m2h":
        encoder, flow, decoder = member["enc_m"], member["flow_m2h"], member["dec_h"]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    sums = np.zeros(out_dim, dtype=np.float64)
    sums_sq = np.zeros(out_dim, dtype=np.float64)
    n_total = 0
    for _, _, block in selected.iter_batches(batch_size):
        x = torch.from_numpy(np.ascontiguousarray(block)).to(device)
        z = flow_step_n(flow, encoder(x), flow_steps)
        mu, _ = decoder(z)
        mu_np = mu.detach().cpu().numpy().astype(np.float64, copy=False)
        sums += mu_np.sum(axis=0)
        sums_sq += np.square(mu_np).sum(axis=0)
        n_total += mu_np.shape[0]
    means = sums / float(n_total)
    return np.maximum(sums_sq / float(n_total) - means**2, 0.0)


def select_translated_variable_genes(
    args: argparse.Namespace,
    source: SelectedDataset,
    direction: str,
    target_genes: np.ndarray,
    human_dim: int,
    mouse_dim: int,
    ensemble_ids: list[int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    top_n = args.top_variable_genes
    target_dim = mouse_dim if direction == "h2m" else human_dim
    if top_n <= 0 or top_n >= target_dim:
        idx = np.arange(target_dim, dtype=np.int64)
        return idx, target_genes[idx]

    variance_sum = np.zeros(target_dim, dtype=np.float64)
    for i, ensemble_id in enumerate(ensemble_ids, start=1):
        print(
            f"[{direction}] estimating translated HVGs with ensemble member "
            f"{ensemble_id} ({i}/{len(ensemble_ids)})",
            flush=True,
        )
        member = load_ensemble_member(args, ensemble_id, human_dim, mouse_dim, device)
        variance_sum += member_translated_gene_variance(
            source,
            member,
            direction,
            device,
            args.batch_size,
            args.flow_steps,
            target_dim,
        )
        del member
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_variance = variance_sum / float(len(ensemble_ids))
    idx = np.argpartition(mean_variance, -top_n)[-top_n:]
    idx = idx[np.argsort(mean_variance[idx])[::-1]].astype(np.int64)
    return idx, target_genes[idx]


@torch.no_grad()
def member_translate(
    selected: SelectedDataset,
    member: dict,
    direction: str,
    device: torch.device,
    batch_size: int,
    flow_steps: int,
    gene_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if direction == "h2m":
        encoder, flow, decoder = member["enc_h"], member["flow_h2m"], member["dec_m"]
    elif direction == "m2h":
        encoder, flow, decoder = member["enc_m"], member["flow_m2h"], member["dec_h"]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    gene_idx_t = torch.as_tensor(gene_idx, dtype=torch.long, device=device)
    out = np.empty((selected.n_samples, len(gene_idx)), dtype=np.float32)
    mean_sq = np.empty(selected.n_samples, dtype=np.float64)
    for start, stop, block in selected.iter_batches(batch_size):
        x = torch.from_numpy(np.ascontiguousarray(block)).to(device)
        z = flow_step_n(flow, encoder(x), flow_steps)
        mu, _ = decoder(z)
        mu = mu.index_select(1, gene_idx_t)
        mu_np = mu.detach().cpu().numpy().astype(np.float32, copy=False)
        out[start:stop] = mu_np
        mean_sq[start:stop] = np.mean(mu_np.astype(np.float64) ** 2, axis=1)
    return out, mean_sq


@torch.no_grad()
def member_reconstruct(
    selected: SelectedDataset,
    member: dict,
    species: str,
    device: torch.device,
    batch_size: int,
    gene_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if species == "human":
        encoder, decoder = member["enc_h"], member["dec_h"]
    elif species == "mouse":
        encoder, decoder = member["enc_m"], member["dec_m"]
    else:
        raise ValueError(f"Unknown species: {species}")

    gene_idx_t = torch.as_tensor(gene_idx, dtype=torch.long, device=device)
    out = np.empty((selected.n_samples, len(gene_idx)), dtype=np.float32)
    mean_sq = np.empty(selected.n_samples, dtype=np.float64)
    for start, stop, block in selected.iter_batches(batch_size):
        x = torch.from_numpy(np.ascontiguousarray(block)).to(device)
        mu, _ = decoder(encoder(x))
        mu = mu.index_select(1, gene_idx_t)
        mu_np = mu.detach().cpu().numpy().astype(np.float32, copy=False)
        out[start:stop] = mu_np
        mean_sq[start:stop] = np.mean(mu_np.astype(np.float64) ** 2, axis=1)
    return out, mean_sq


def finalize_ensemble_mean(sum_expr: np.ndarray, sum_mean_sq: np.ndarray, n_models: int):
    mean_expr = sum_expr / float(n_models)
    mean_expr_sq = np.mean(mean_expr.astype(np.float64) ** 2, axis=1)
    expected_sq = sum_mean_sq / float(n_models)
    ensemble_sd = np.sqrt(np.maximum(expected_sq - mean_expr_sq, 0.0))
    return mean_expr.astype(np.float32, copy=False), ensemble_sd


def ensemble_translate_and_reference(
    args: argparse.Namespace,
    source: SelectedDataset,
    target: SelectedDataset,
    direction: str,
    target_species: str,
    human_dim: int,
    mouse_dim: int,
    ensemble_ids: list[int],
    device: torch.device,
    gene_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    target_dim = len(gene_idx)
    translated_sum = np.zeros((source.n_samples, target_dim), dtype=np.float32)
    translated_sum_mean_sq = np.zeros(source.n_samples, dtype=np.float64)

    reference_sum = None
    reference_sum_mean_sq = None
    if args.target_reference == "reconstructed":
        reference_sum = np.zeros((target.n_samples, target_dim), dtype=np.float32)
        reference_sum_mean_sq = np.zeros(target.n_samples, dtype=np.float64)

    for i, ensemble_id in enumerate(ensemble_ids, start=1):
        print(f"[{direction}] loading ensemble member {ensemble_id} ({i}/{len(ensemble_ids)})", flush=True)
        member = load_ensemble_member(args, ensemble_id, human_dim, mouse_dim, device)

        translated, translated_mean_sq = member_translate(
            source,
            member,
            direction,
            device,
            args.batch_size,
            args.flow_steps,
            gene_idx,
        )
        translated_sum += translated
        translated_sum_mean_sq += translated_mean_sq

        if args.target_reference == "reconstructed":
            reference, reference_mean_sq = member_reconstruct(
                target,
                member,
                target_species,
                device,
                args.batch_size,
                gene_idx,
            )
            reference_sum += reference
            reference_sum_mean_sq += reference_mean_sq

        del member
        if device.type == "cuda":
            torch.cuda.empty_cache()

    translated_mean, translated_sd = finalize_ensemble_mean(
        translated_sum,
        translated_sum_mean_sq,
        len(ensemble_ids),
    )

    if args.target_reference == "reconstructed":
        reference_mean, reference_sd = finalize_ensemble_mean(
            reference_sum,
            reference_sum_mean_sq,
            len(ensemble_ids),
        )
    else:
        reference_mean = target.to_numpy(args.batch_size, cols=gene_idx)
        reference_sd = None

    return translated_mean, translated_sd, reference_mean, reference_sd


def run_umap(X: np.ndarray, args: argparse.Namespace):
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "The package umap-learn is required. Install it in the cluster "
            "environment, for example: pip install umap-learn"
        ) from exc

    X_centered = X.astype(np.float32, copy=True)
    X_centered -= X_centered.mean(axis=0, keepdims=True)
    n_components = min(args.pca_components, X_centered.shape[0] - 1, X_centered.shape[1])
    if n_components < 2:
        raise ValueError("Not enough samples/features for PCA + UMAP.")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=args.seed)
    X_pca = pca.fit_transform(X_centered)

    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
        n_components=2,
        low_memory=True,
    )
    embedding = reducer.fit_transform(X_pca)
    return embedding, pca.explained_variance_ratio_


def plot_umap(coords: pd.DataFrame, direction_label: str, out_path: Path):
    colors = {
        "real_target_reconstructed": "#4C78A8",
        "real_target_raw": "#4C78A8",
        "translated_source_consensus": "#E15759",
    }
    labels = {
        "real_target_reconstructed": "Real target, ensemble reconstructed",
        "real_target_raw": "Real target, raw processed expression",
        "translated_source_consensus": "Source translated to target",
    }
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    for point_type, sub in coords.groupby("point_type", sort=False):
        ax.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            s=9,
            alpha=0.35 if len(coords) > 50000 else 0.55,
            c=colors.get(point_type, "grey50"),
            label=labels.get(point_type, point_type),
            linewidths=0,
            rasterized=True,
        )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(direction_label)
    ax.legend(frameon=False, loc="best", markerscale=2.0)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def process_direction(
    args: argparse.Namespace,
    direction: str,
    source: SelectedDataset,
    target: SelectedDataset,
    target_genes: np.ndarray,
    human_dim: int,
    mouse_dim: int,
    ensemble_ids: list[int],
    device: torch.device,
):
    target_species = target.species
    source_species = source.species
    raw_gene_idx, _ = select_raw_target_variable_genes(
        target,
        target_genes.astype(str),
        args.top_variable_genes,
        args.batch_size,
    )
    translated_gene_idx, _ = select_translated_variable_genes(
        args,
        source,
        direction,
        target_genes.astype(str),
        human_dim,
        mouse_dim,
        ensemble_ids,
        device,
    )
    gene_idx = np.array(
        list(dict.fromkeys(np.concatenate([raw_gene_idx, translated_gene_idx]).tolist())),
        dtype=np.int64,
    )
    selected_genes = target_genes.astype(str)[gene_idx]
    translated, translated_sd, reference, reference_sd = ensemble_translate_and_reference(
        args,
        source,
        target,
        direction,
        target_species,
        human_dim,
        mouse_dim,
        ensemble_ids,
        device,
        gene_idx,
    )

    point_type = (
        "real_target_reconstructed"
        if args.target_reference == "reconstructed"
        else "real_target_raw"
    )
    metadata = pd.concat(
        [
            target.metadata(point_type, f"{target_species}_expression", reference_sd),
            source.metadata(
                "translated_source_consensus",
                f"{target_species}_expression",
                translated_sd,
            ),
        ],
        ignore_index=True,
    )
    metadata["source_species"] = np.where(
        metadata["point_type"].eq("translated_source_consensus"),
        source_species,
        metadata["original_species"],
    )
    metadata["direction"] = direction
    metadata["ensemble_ids"] = ",".join(str(x) for x in ensemble_ids)
    metadata["target_reference"] = args.target_reference

    combined_selected = np.vstack([reference, translated]).astype(np.float32, copy=False)
    embedding, explained = run_umap(combined_selected, args)

    coords = metadata.copy()
    coords["UMAP1"] = embedding[:, 0]
    coords["UMAP2"] = embedding[:, 1]

    prefix = "human_to_mouse_space" if direction == "h2m" else "mouse_to_human_space"
    coords.to_csv(args.out_dir / f"{prefix}_umap_coordinates.csv", index=False)
    raw_set = set(raw_gene_idx.tolist())
    translated_set = set(translated_gene_idx.tolist())
    pd.DataFrame(
        {
            "gene": selected_genes,
            "target_gene_index": gene_idx,
            "selected_from_raw_target": [idx in raw_set for idx in gene_idx.tolist()],
            "selected_from_translated_source": [
                idx in translated_set for idx in gene_idx.tolist()
            ],
        }
    ).to_csv(
        args.out_dir / f"{prefix}_selected_variable_genes.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "pc": np.arange(1, len(explained) + 1),
            "explained_variance_ratio": explained,
        }
    ).to_csv(args.out_dir / f"{prefix}_pca_explained_variance.csv", index=False)

    direction_label = (
        "Human samples translated into mouse expression space"
        if direction == "h2m"
        else "Mouse samples translated into human expression space"
    )
    plot_umap(coords, direction_label, args.out_dir / f"{prefix}_umap.png")

    if args.save_embedding_matrix:
        np.save(args.out_dir / f"{prefix}_umap_input_matrix.npy", combined_selected)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preproc_dir", type=Path, default=PREPROC_DIR)
    parser.add_argument("--model_dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--ensemble_ids", default="0-9")
    parser.add_argument("--full_model_prefix", default="full_ensemble")
    parser.add_argument(
        "--n_human",
        default="all",
        help="Number of human samples to plot, or 'all' for the full preprocessed human matrix.",
    )
    parser.add_argument(
        "--n_mouse",
        default="all",
        help="Number of mouse samples to plot, or 'all' for the full preprocessed mouse matrix.",
    )
    parser.add_argument("--include_test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test_fraction", type=float, default=0.25)
    parser.add_argument("--max_test_samples", type=int, default=1000)
    parser.add_argument("--target_reference", choices=["reconstructed", "raw"], default="reconstructed")
    parser.add_argument("--top_variable_genes", type=int, default=3000)
    parser.add_argument("--pca_components", type=int, default=50)
    parser.add_argument("--umap_neighbors", type=int, default=50)
    parser.add_argument("--umap_min_dist", type=float, default=0.25)
    parser.add_argument("--umap_metric", default="euclidean")
    parser.add_argument("--flow_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8128)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--save_embedding_matrix", action="store_true")
    parser.add_argument("--skip_h2m", action="store_true")
    parser.add_argument("--skip_m2h", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ensemble_ids = parse_int_ranges(args.ensemble_ids)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}", flush=True)

    human_genes = np.load(require_file(args.preproc_dir / "human_genes.npy"), allow_pickle=True).astype(str)
    mouse_genes = np.load(require_file(args.preproc_dir / "mouse_genes.npy"), allow_pickle=True).astype(str)
    human_dim = len(human_genes)
    mouse_dim = len(mouse_genes)

    human = sample_species_dataset(
        args.preproc_dir,
        "human",
        args.n_human,
        args.include_test,
        args.test_fraction,
        args.max_test_samples,
        args.seed + 101,
    )
    mouse = sample_species_dataset(
        args.preproc_dir,
        "mouse",
        args.n_mouse,
        args.include_test,
        args.test_fraction,
        args.max_test_samples,
        args.seed + 202,
    )

    config = {
        "ensemble_ids": ensemble_ids,
        "n_human_selected": human.n_samples,
        "n_mouse_selected": mouse.n_samples,
        "human_dim": human_dim,
        "mouse_dim": mouse_dim,
        "target_reference": args.target_reference,
        "top_variable_genes": args.top_variable_genes,
        "variable_gene_strategy": (
            "union of top_variable_genes raw target HVGs and top_variable_genes "
            "translated-source HVGs; translated HVGs use the mean per-gene variance "
            "across ensemble members"
        ),
        "pca_preprocessing": "gene-wise mean centering only; no variance scaling/z-scoring",
        "pca_components": args.pca_components,
        "umap_neighbors": args.umap_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "umap_metric": args.umap_metric,
        "include_test": args.include_test,
        "test_fraction": args.test_fraction,
        "max_test_samples": args.max_test_samples,
        "seed": args.seed,
    }
    with open(args.out_dir / "umap_run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    if not args.skip_h2m:
        process_direction(
            args,
            "h2m",
            source=human,
            target=mouse,
            target_genes=mouse_genes,
            human_dim=human_dim,
            mouse_dim=mouse_dim,
            ensemble_ids=ensemble_ids,
            device=device,
        )

    if not args.skip_m2h:
        process_direction(
            args,
            "m2h",
            source=mouse,
            target=human,
            target_genes=human_genes,
            human_dim=human_dim,
            mouse_dim=mouse_dim,
            ensemble_ids=ensemble_ids,
            device=device,
        )

    print(f"Wrote ensemble UMAP outputs to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
