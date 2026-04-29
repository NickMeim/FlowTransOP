#!/usr/bin/env python3
"""
Train ONLY the mouse->human flow for a single fold, reusing the trained encoders
from fold_{f}_normal.pt and fold_{f}_permuted.pt.
"""
import argparse
from pathlib import Path
import torch
import numpy as np
from models import SimpleEncoder, Flow, ElementWiseLinear
from trainingUtils import train_RNAseq_flowMatch_fold, validate_RNAseq_flowMatch_fold
from utility import *
from transact_utility_gpu import *
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import logging
from logging import FileHandler
import warnings
warnings.filterwarnings('ignore', message='.*ks_2samp.*')

DATA_DIR    = Path("../archs4")
PREPROC_DIR = DATA_DIR / "preprocessed"
MODEL_DIR   = Path("../archs4/models")


# --- Same lazy data wrappers as the main training script ---
class LazyMatrix:
    def __init__(self, mat_path: Path, row_index: np.ndarray = None):
        self._mat = np.load(mat_path, mmap_mode='r')
        self._row_index = row_index
    @property
    def shape(self):
        n = self._row_index.shape[0] if self._row_index is not None else self._mat.shape[0]
        return (n, self._mat.shape[1])
    def __len__(self): return self.shape[0]
    def __getitem__(self, key):
        row_key, col_key = (key if isinstance(key, tuple) else (key, slice(None)))
        phys = self._row_index[row_key] if self._row_index is not None else row_key
        block = np.asarray(self._mat[phys])
        if not (isinstance(col_key, slice) and col_key == slice(None)):
            block = block[:, col_key]
        return torch.from_numpy(np.ascontiguousarray(block)).float()


def load_fold(species, fold):
    train_i = np.load(PREPROC_DIR / f"{species}_fold{fold}_train_idx.npy")
    val_i   = np.load(PREPROC_DIR / f"{species}_fold{fold}_val_idx.npy")
    mat     = PREPROC_DIR / f"{species}_X.npy"
    return LazyMatrix(mat, row_index=train_i), LazyMatrix(mat, row_index=val_i)


class PermutedLazy:
    def __init__(self, base, col_perm): self.base, self.cp = base, col_perm
    @property
    def shape(self): return self.base.shape
    def __len__(self): return len(self.base)
    def __getitem__(self, key):
        t = self.base[key]
        return t[:, torch.as_tensor(self.cp, dtype=torch.long)]


@torch.no_grad()
def encode_all(encoder, X_lazy, device, bs=4096):
    encoder.eval()
    N = len(X_lazy)
    out = []
    for c0 in range(0, N, bs):
        c1 = min(c0 + bs, N)
        X = X_lazy[np.arange(c0, c1), :].to(device)
        out.append(encoder(X).cpu())
    return torch.cat(out, dim=0)


def _activation(name):
    return {'LeakyReLU': torch.nn.LeakyReLU(0.01),
            'ReLU':      torch.nn.ReLU(),
            'ELU':       torch.nn.ELU(),
            'Sigmoid':   torch.nn.Sigmoid()}[name]


def rebuild_models_from_ckpt(ckpt, Gh, Gm, device):
    """Reconstruct encoder architectures from a checkpoint and load weights.
    Gene dims are passed in (read from the preprocessed data) since the forward
    training script no longer stores them in the checkpoint."""
    a  = ckpt['args']
    enc_act = _activation(a['encoder_activation'])

    encoder_human = torch.nn.Sequential(
        ElementWiseLinear(Gh),
        SimpleEncoder(Gh, a['encoder_1_hiddens'], a['latent_dim'],
                      dropRate=a['dropout_encoder'], bn=a['bn_encoder'],
                      activation=enc_act, dropIn=a['dropout_input_encoder'],
                      dtype=torch.float)).to(device)
    encoder_mouse = torch.nn.Sequential(
        ElementWiseLinear(Gm),
        SimpleEncoder(Gm, a['encoder_2_hiddens'], a['latent_dim'],
                      dropRate=a['dropout_encoder'], bn=a['bn_encoder'],
                      activation=enc_act, dropIn=a['dropout_input_encoder'],
                      dtype=torch.float)).to(device)

    encoder_human.load_state_dict(ckpt['encoder_human'])
    encoder_mouse.load_state_dict(ckpt['encoder_mouse'])

    for m in (encoder_human, encoder_mouse):
        m.eval()
    return encoder_human, encoder_mouse, a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs (defaults to whatever the original run used).")
    args_cli = parser.parse_args()

    log_file = f'logs/ARCHS4_fold_{args_cli.fold}_m2h.log'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = FileHandler(log_file, mode='a')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    print2log = logger.info

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print2log(f"Using device: {device}")

    # Load data
    print2log(f"Loading preprocessed fold {args_cli.fold} ...")
    X_human, X_human_val = load_fold("human", args_cli.fold)
    X_mouse, X_mouse_val = load_fold("mouse", args_cli.fold)
    print2log(f"Human train: {X_human.shape}, val: {X_human_val.shape}")
    print2log(f"Mouse train: {X_mouse.shape}, val: {X_mouse_val.shape}")
    Gh = X_human.shape[1]
    Gm = X_mouse.shape[1]

    # =====================================================================
    # NORMAL: load encoders, train m2h flow
    # =====================================================================
    print2log("\n=== Training NORMAL m2h flow ===")
    ckpt_normal = torch.load(MODEL_DIR / f"fold_{args_cli.fold}_normal.pt",
                             map_location=device, weights_only=False)
    enc_h, enc_m, a = rebuild_models_from_ckpt(ckpt_normal, Gh, Gm, device)
    epochs = args_cli.epochs if args_cli.epochs is not None else a['epochs']

    # Reproducibility: use the seed the forward run used (falls back to 42 if absent)
    seed = int(a.get('seed', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build model_params dict from saved args (needed by train_RNAseq_flowMatch_fold)
    model_params = {**a,
                    'encoder_activation': _activation(a['encoder_activation'])}

    # Encode latents
    Z_human = encode_all(enc_h, X_human, device).to(device)
    Z_mouse = encode_all(enc_m, X_mouse, device).to(device)

    # Build the reverse flow
    flow_m2h = Flow(a['latent_dim'], a['latent_dim']//2, dtype=torch.float).to(device)

    # Train flow with translation_direction='2 to 1'
    z_m2h, flow_m2h = train_RNAseq_flowMatch_fold(
        model_params, device,
        X_human, X_mouse,
        Z_human, Z_mouse,
        flow_m2h,
        a['batch_size'], a['batch_size'], epochs,
        translation_direction='2 to 1',
        plot_label=f'm2h_fold{args_cli.fold}')

    torch.save({
        'flow_m2h': flow_m2h.state_dict(),
        'args': a,
    }, MODEL_DIR / f"fold_{args_cli.fold}_normal_m2h.pt")

    # Evaluate on val
    flow_m2h.eval()
    with torch.no_grad():
        z_m2h_val = validate_RNAseq_flowMatch_fold(
            device,
            X_human_val, X_mouse_val,
            enc_h, enc_m, flow_m2h,
            translation_direction='2 to 1')

    np.save(MODEL_DIR / f"fold_{args_cli.fold}_z_m2h_train.npy", z_m2h.cpu().numpy())
    np.save(MODEL_DIR / f"fold_{args_cli.fold}_z_m2h_val.npy",   z_m2h_val.cpu().numpy())

    del enc_h, enc_m, Z_human, Z_mouse, flow_m2h, ckpt_normal
    torch.cuda.empty_cache()

    # =====================================================================
    # PERMUTED: load encoders + perm indices, train m2h flow on permuted train
    # =====================================================================
    print2log("\n=== Training PERMUTED m2h flow ===")
    ckpt_perm = torch.load(MODEL_DIR / f"fold_{args_cli.fold}_permuted.pt",
                           map_location=device, weights_only=False)
    enc_h_p, enc_m_p, a_p = rebuild_models_from_ckpt(ckpt_perm, Gh, Gm, device)

    # Reuse the EXACT permutations the permuted models were trained with
    perm_idx_human = ckpt_perm['perm_idx_human']
    perm_idx_mouse = ckpt_perm['perm_idx_mouse']
    print2log(f"Loaded saved perm indices — human:{perm_idx_human.shape}, mouse:{perm_idx_mouse.shape}")

    X_human_permuted = PermutedLazy(X_human, perm_idx_human)
    X_mouse_permuted = PermutedLazy(X_mouse, perm_idx_mouse)
    # NOTE: val stays unpermuted, matching how train_ARCHS4_fold.py was fixed.

    Z_human_perm = encode_all(enc_h_p, X_human_permuted, device).to(device)
    Z_mouse_perm = encode_all(enc_m_p, X_mouse_permuted, device).to(device)

    flow_m2h_perm = Flow(a_p['latent_dim'], a_p['latent_dim']//2, dtype=torch.float).to(device)
    model_params_p = {**a_p,
                      'encoder_activation': _activation(a_p['encoder_activation'])}

    z_m2h_perm, flow_m2h_perm = train_RNAseq_flowMatch_fold(
        model_params_p, device,
        X_human_permuted, X_mouse_permuted,
        Z_human_perm, Z_mouse_perm,
        flow_m2h_perm,
        a_p['batch_size'], a_p['batch_size'], epochs,
        translation_direction='2 to 1',
        plot_label=f'permuted_m2h_fold{args_cli.fold}')

    torch.save({
        'flow_m2h':       flow_m2h_perm.state_dict(),
        'args':           a_p,
        'perm_idx_human': perm_idx_human,
        'perm_idx_mouse': perm_idx_mouse,
    }, MODEL_DIR / f"fold_{args_cli.fold}_permuted_m2h.pt")

    flow_m2h_perm.eval()
    with torch.no_grad():
        z_m2h_val_perm = validate_RNAseq_flowMatch_fold(
            device,
            X_human_val, X_mouse_val,                # unpermuted val (same convention as forward run)
            enc_h_p, enc_m_p, flow_m2h_perm,
            translation_direction='2 to 1')

    np.save(MODEL_DIR / f"fold_{args_cli.fold}_z_m2h_train_perm.npy", z_m2h_perm.cpu().numpy())
    np.save(MODEL_DIR / f"fold_{args_cli.fold}_z_m2h_val_perm.npy",   z_m2h_val_perm.cpu().numpy())

    print2log(f"\n✓ Fold {args_cli.fold} m2h complete!")


if __name__ == "__main__":
    main()