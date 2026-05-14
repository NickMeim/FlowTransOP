#!/usr/bin/env python3
"""
Train one full-data ARCHS4 FlowTransOP ensemble member.

This mirrors train_ARCHS4_fold.py, but uses fold 0 train + validation rows plus
the held-out liver test matrix, trains both h2m and m2h flows, and saves only
the model components. No permuted/shuffled controls and no latent arrays are
written.
"""

from __future__ import annotations

import argparse
import logging
import random
import warnings
from logging import FileHandler
from pathlib import Path

import numpy as np
import torch

from models import ElementWiseLinear, Flow, SimpleEncoder, VarDecoder
from trainingUtils import train_RNAseq_AE_fold_gauss, train_RNAseq_flowMatch_fold

warnings.filterwarnings("ignore", message=".*ks_2samp.*")


DATA_DIR = Path("../archs4")
PREPROC_DIR = DATA_DIR / "preprocessed"
MODEL_DIR = DATA_DIR / "models"


class LazyMatrix:
    """Sample-major memmap wrapper returning float32 torch tensors."""

    def __init__(self, mat_path: Path, row_index: np.ndarray | None = None):
        self._mat = np.load(mat_path, mmap_mode="r")
        self._row_index = row_index

    @property
    def shape(self):
        n = self._row_index.shape[0] if self._row_index is not None else self._mat.shape[0]
        return (n, self._mat.shape[1])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        row_key, col_key = (key if isinstance(key, tuple) else (key, slice(None)))
        phys = self._row_index[row_key] if self._row_index is not None else row_key
        block = np.asarray(self._mat[phys])
        if not (isinstance(col_key, slice) and col_key == slice(None)):
            block = block[:, col_key]
        return torch.from_numpy(np.ascontiguousarray(block)).float()


class ConcatLazyMatrix:
    """Concatenate LazyMatrix-like objects along rows without materializing them."""

    def __init__(self, parts):
        if not parts:
            raise ValueError("ConcatLazyMatrix requires at least one part.")
        n_features = {part.shape[1] for part in parts}
        if len(n_features) != 1:
            raise ValueError(f"All matrix parts must have the same feature count: {n_features}")
        self.parts = list(parts)
        self.offsets = np.cumsum([0] + [part.shape[0] for part in self.parts])
        self._shape = (int(self.offsets[-1]), int(self.parts[0].shape[1]))

    @property
    def shape(self):
        return self._shape

    def __len__(self):
        return self._shape[0]

    def __getitem__(self, key):
        row_key, col_key = (key if isinstance(key, tuple) else (key, slice(None)))
        if not (isinstance(col_key, slice) and col_key == slice(None)):
            raise NotImplementedError("ConcatLazyMatrix currently supports full-column reads only.")

        if isinstance(row_key, slice):
            rows = np.arange(self._shape[0])[row_key]
        else:
            rows = np.asarray(row_key)
        scalar = rows.ndim == 0
        rows = rows.reshape(-1).astype(np.int64)

        out = torch.empty((rows.shape[0], self._shape[1]), dtype=torch.float32)
        for part_i, part in enumerate(self.parts):
            start, stop = self.offsets[part_i], self.offsets[part_i + 1]
            mask = (rows >= start) & (rows < stop)
            if np.any(mask):
                local_rows = rows[mask] - start
                out[torch.from_numpy(np.where(mask)[0])] = part[local_rows, :]

        return out[0] if scalar else out


def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return path


def load_full_species(species: str, fold: int, preproc_dir: Path, include_liver_test: bool):
    train_i = np.load(require_file(preproc_dir / f"{species}_fold{fold}_train_idx.npy"))
    val_i = np.load(require_file(preproc_dir / f"{species}_fold{fold}_val_idx.npy"))
    train_val_i = np.concatenate([train_i, val_i]).astype(np.int64)

    parts = [LazyMatrix(require_file(preproc_dir / f"{species}_X.npy"), row_index=train_val_i)]
    test_path = preproc_dir / f"{species}_test_X.npy"
    if include_liver_test:
        parts.append(LazyMatrix(require_file(test_path)))
    elif test_path.exists():
        logging.info("Skipping %s because --include_liver_test is false.", test_path)

    return ConcatLazyMatrix(parts)


def activation(name):
    return {
        "LeakyReLU": torch.nn.LeakyReLU(0.01),
        "ReLU": torch.nn.ReLU(),
        "ELU": torch.nn.ELU(),
        "Sigmoid": torch.nn.Sigmoid(),
    }[name]


def build_models(model_params, human_dim, mouse_dim, device):
    encoder_human = torch.nn.Sequential(
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
    encoder_mouse = torch.nn.Sequential(
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
    decoder_human = VarDecoder(
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
    decoder_mouse = VarDecoder(
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
    flow_h2m = Flow(model_params["latent_dim"], model_params["latent_dim"] // 2, dtype=torch.float).to(device)
    flow_m2h = Flow(model_params["latent_dim"], model_params["latent_dim"] // 2, dtype=torch.float).to(device)
    return encoder_human, encoder_mouse, decoder_human, decoder_mouse, flow_h2m, flow_m2h


@torch.no_grad()
def encode_all(encoder, X_lazy, device, bs=4096):
    encoder.eval()
    out = []
    for c0 in range(0, len(X_lazy), bs):
        c1 = min(c0 + bs, len(X_lazy))
        x = X_lazy[np.arange(c0, c1), :].to(device)
        out.append(encoder(x).cpu())
    return torch.cat(out, dim=0)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_args(args, effective_seed):
    out = vars(args).copy()
    out["effective_seed"] = int(effective_seed)
    out["preproc_dir"] = str(out["preproc_dir"])
    out["model_dir"] = str(out["model_dir"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_id", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--preproc_dir", type=Path, default=PREPROC_DIR)
    parser.add_argument("--model_dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--include_liver_test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--seed_stride", type=int, default=1000003)
    parser.add_argument("--enc_l2_reg", type=float, default=0.001)
    parser.add_argument("--dec_l2_reg", type=float, default=0.001)
    parser.add_argument("--encoding_lr", type=float, default=0.001)
    parser.add_argument("--schedule_step_enc", type=int, default=20)
    parser.add_argument("--gamma_enc", type=float, default=0.8)
    parser.add_argument("--autoencoder_wd", type=float, default=0.0)
    parser.add_argument("--encoder_1_hiddens", type=int, nargs="+", default=[4096, 2048, 1024, 512])
    parser.add_argument("--encoder_2_hiddens", type=int, nargs="+", default=[4096, 2048, 1024, 512])
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--decoder_1_hiddens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--decoder_2_hiddens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--dropout_decoder", type=float, default=0.2)
    parser.add_argument("--dropout_encoder", type=float, default=0.2)
    parser.add_argument("--bn_decoder", type=float, default=0.6)
    parser.add_argument("--bn_encoder", type=float, default=0.6)
    parser.add_argument("--dropout_input_encoder", type=float, default=0.5)
    parser.add_argument("--dropout_input_decoder", type=float, default=0.2)
    parser.add_argument(
        "--encoder_activation",
        choices=["LeakyReLU", "ReLU", "ELU", "Sigmoid"],
        default="ELU",
    )
    parser.add_argument(
        "--decoder_activation",
        choices=["LeakyReLU", "ReLU", "ELU", "Sigmoid"],
        default="ELU",
    )
    parser.add_argument("--flow_lambda", type=float, default=1.0)
    parser.add_argument("--conditional_flow_lambda", type=float, default=1e-3)
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("training_plots").mkdir(exist_ok=True)

    log_file = Path("logs") / f"ARCHS4_full_ensemble_{args.ensemble_id}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = FileHandler(log_file, mode="a")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    print2log = logger.info

    effective_seed = int(args.base_seed + args.ensemble_id * args.seed_stride)
    set_seeds(effective_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print2log(f"Using device: {device}")
    print2log(f"Ensemble id: {args.ensemble_id}; effective seed: {effective_seed}")

    print2log(f"Loading fold {args.fold} train+val plus liver test matrices...")
    X_human = load_full_species("human", args.fold, args.preproc_dir, args.include_liver_test)
    X_mouse = load_full_species("mouse", args.fold, args.preproc_dir, args.include_liver_test)
    print2log(f"Human full training matrix: {X_human.shape}")
    print2log(f"Mouse full training matrix: {X_mouse.shape}")

    model_params = {
        "encoder_1_hiddens": args.encoder_1_hiddens,
        "encoder_2_hiddens": args.encoder_2_hiddens,
        "latent_dim": args.latent_dim,
        "decoder_1_hiddens": args.decoder_1_hiddens,
        "decoder_2_hiddens": args.decoder_2_hiddens,
        "dropout_decoder": args.dropout_decoder,
        "dropout_encoder": args.dropout_encoder,
        "encoder_activation": activation(args.encoder_activation),
        "decoder_activation": activation(args.decoder_activation),
        "bn_encoder": args.bn_encoder,
        "bn_decoder": args.bn_decoder,
        "dropout_input_encoder": args.dropout_input_encoder,
        "dropout_input_decoder": args.dropout_input_decoder,
        "encoding_lr": args.encoding_lr,
        "schedule_step_enc": args.schedule_step_enc,
        "gamma_enc": args.gamma_enc,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "enc_l2_reg": args.enc_l2_reg,
        "dec_l2_reg": args.dec_l2_reg,
        "autoencoder_wd": args.autoencoder_wd,
        "flow_lambda": args.flow_lambda,
        "conditional_flow_lambda": args.conditional_flow_lambda,
    }

    encoder_human, encoder_mouse, decoder_human, decoder_mouse, flow_h2m, flow_m2h = build_models(
        model_params, X_human.shape[1], X_mouse.shape[1], device
    )

    print2log("Training human autoencoder...")
    _, decoder_human, encoder_human = train_RNAseq_AE_fold_gauss(
        model_params,
        device,
        X_human,
        decoder_human,
        encoder_human,
        model_params["batch_size"],
        model_params["epochs"],
        evaluate=False,
        plot_label=f"full_ens{args.ensemble_id}_human",
    )

    print2log("Training mouse autoencoder...")
    _, decoder_mouse, encoder_mouse = train_RNAseq_AE_fold_gauss(
        model_params,
        device,
        X_mouse,
        decoder_mouse,
        encoder_mouse,
        model_params["batch_size"],
        model_params["epochs"],
        evaluate=False,
        plot_label=f"full_ens{args.ensemble_id}_mouse",
    )

    print2log("Encoding full human and mouse matrices...")
    Z_human = encode_all(encoder_human, X_human, device, bs=args.batch_size).to(device)
    Z_mouse = encode_all(encoder_mouse, X_mouse, device, bs=args.batch_size).to(device)

    print2log("Training human->mouse flow...")
    _, flow_h2m = train_RNAseq_flowMatch_fold(
        model_params,
        device,
        X_human,
        X_mouse,
        Z_human,
        Z_mouse,
        flow_h2m,
        model_params["batch_size"],
        model_params["batch_size"],
        model_params["epochs"],
        translation_direction="1 to 2",
        plot_label=f"full_ens{args.ensemble_id}_h2m",
    )

    print2log("Training mouse->human flow...")
    _, flow_m2h = train_RNAseq_flowMatch_fold(
        model_params,
        device,
        X_human,
        X_mouse,
        Z_human,
        Z_mouse,
        flow_m2h,
        model_params["batch_size"],
        model_params["batch_size"],
        model_params["epochs"],
        translation_direction="2 to 1",
        plot_label=f"full_ens{args.ensemble_id}_m2h",
    )

    ckpt_args = save_args(args, effective_seed)
    normal_path = args.model_dir / f"full_ensemble_{args.ensemble_id}_normal.pt"
    m2h_path = args.model_dir / f"full_ensemble_{args.ensemble_id}_normal_m2h.pt"
    torch.save(
        {
            "encoder_human": encoder_human.state_dict(),
            "encoder_mouse": encoder_mouse.state_dict(),
            "decoder_human": decoder_human.state_dict(),
            "decoder_mouse": decoder_mouse.state_dict(),
            "flow_h2m": flow_h2m.state_dict(),
            "flow_m2h": flow_m2h.state_dict(),
            "args": ckpt_args,
        },
        normal_path,
    )
    torch.save(
        {
            "flow_m2h": flow_m2h.state_dict(),
            "args": ckpt_args,
        },
        m2h_path,
    )

    print2log(f"Wrote: {normal_path}")
    print2log(f"Wrote: {m2h_path}")
    print2log("Full-data ensemble training complete.")


if __name__ == "__main__":
    main()
