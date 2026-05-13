#!/usr/bin/env python3
"""
Evaluate FlowTransOP cross-species translation for a single fold.

Loads:
  - fold_{f}_normal.pt        : enc_h, enc_m, dec_h, dec_m, flow_h2m
  - fold_{f}_permuted.pt      : same + perm_idx_*
  - fold_{f}_normal_m2h.pt    : flow_m2h
  - fold_{f}_permuted_m2h.pt  : flow_m2h + perm_idx_*

Runs:
  1. Cycle consistency (human and mouse, 4 ablation modes each)
  2. Orthologue-mediated comparison (h2m and m2h, 4 modes each)
  3. MMD on latents

Convention for ablation modes (cycle consistency only):
  Stages of human cycle:  enc_h â†’ flow_h2m â†’ flow_m2h â†’ dec_h
  Stages of mouse cycle:  enc_m â†’ flow_m2h â†’ flow_h2m â†’ dec_m

  - FlowTransOP       : all 4 stages from _normal.pt
  - permuted_both     : all 4 stages from _permuted.pt
  - permuted_<HOME>   : encoder + decoder of the cycle's home species permuted;
                        flows from _normal.pt
  - permuted_<THROUGH>: flows permuted; home enc/dec normal

  So "permuted_human" means different things depending on which cycle:
  - In human cycle (home=human)  : enc_h, dec_h are permuted, flows normal
  - In mouse cycle (through=human): both flows are permuted, enc_m/dec_m normal

For orthologue eval and MMD, modes are simpler combinations of the
3 component-stages used (encoder, flow, decoder).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging
from logging import FileHandler
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from models import VarDecoder, SimpleEncoder, Flow, ElementWiseLinear

DATA_DIR    = Path("../archs4")
PREPROC_DIR = DATA_DIR / "preprocessed"
MODEL_DIR   = DATA_DIR / "models"
EVAL_DIR    = DATA_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- data wrappers -----------------------------

class LazyMatrix:
    def __init__(self, mat_path, row_index=None):
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


def load_val(species, fold):
    val_i = np.load(PREPROC_DIR / f"{species}_fold{fold}_val_idx.npy")
    return LazyMatrix(PREPROC_DIR / f"{species}_X.npy", row_index=val_i)


def load_genes(species):
    return np.load(PREPROC_DIR / f"{species}_genes.npy", allow_pickle=True)


# ----------------------------- model builders -----------------------------

def _activation(name):
    return {'LeakyReLU': torch.nn.LeakyReLU(0.01),
            'ReLU':      torch.nn.ReLU(),
            'ELU':       torch.nn.ELU(),
            'Sigmoid':   torch.nn.Sigmoid()}[name]


def build_models(a, Gh, Gm, device):
    enc_act = _activation(a['encoder_activation'])
    dec_act = _activation(a['decoder_activation'])
    enc_h = torch.nn.Sequential(
        ElementWiseLinear(Gh),
        SimpleEncoder(Gh, a['encoder_1_hiddens'], a['latent_dim'],
                      dropRate=a['dropout_encoder'], bn=a['bn_encoder'],
                      activation=enc_act, dropIn=a['dropout_input_encoder'],
                      dtype=torch.float)).to(device)
    enc_m = torch.nn.Sequential(
        ElementWiseLinear(Gm),
        SimpleEncoder(Gm, a['encoder_2_hiddens'], a['latent_dim'],
                      dropRate=a['dropout_encoder'], bn=a['bn_encoder'],
                      activation=enc_act, dropIn=a['dropout_input_encoder'],
                      dtype=torch.float)).to(device)
    dec_h = VarDecoder(a['latent_dim'], a['decoder_1_hiddens'], Gh,
                       dropRate=a['dropout_decoder'], bn=a['bn_decoder'],
                       activation=dec_act, dropIn=a['dropout_input_decoder'],
                       loss='gauss', dtype=torch.float).to(device)
    dec_m = VarDecoder(a['latent_dim'], a['decoder_2_hiddens'], Gm,
                       dropRate=a['dropout_decoder'], bn=a['bn_decoder'],
                       activation=dec_act, dropIn=a['dropout_input_decoder'],
                       loss='gauss', dtype=torch.float).to(device)
    flow = Flow(a['latent_dim'], a['latent_dim']//2, dtype=torch.float).to(device)
    return enc_h, enc_m, dec_h, dec_m, flow


def load_all_components(fold, device):
    """Returns dict: {'normal': {...}, 'permuted': {...}, 'args': ..., 'Gh':, 'Gm':, ...}"""
    ckpt_n     = torch.load(MODEL_DIR / f"fold_{fold}_normal.pt",       map_location=device, weights_only=False)
    ckpt_p     = torch.load(MODEL_DIR / f"fold_{fold}_permuted.pt",     map_location=device, weights_only=False)
    ckpt_n_m2h = torch.load(MODEL_DIR / f"fold_{fold}_normal_m2h.pt",   map_location=device, weights_only=False)
    ckpt_p_m2h = torch.load(MODEL_DIR / f"fold_{fold}_permuted_m2h.pt", map_location=device, weights_only=False)
    a = ckpt_n['args']

    val_h = load_val("human", fold)
    val_m = load_val("mouse", fold)
    Gh, Gm = val_h.shape[1], val_m.shape[1]

    # Normal
    enc_h_n, enc_m_n, dec_h_n, dec_m_n, flow_h2m_n = build_models(a, Gh, Gm, device)
    enc_h_n.load_state_dict(ckpt_n['encoder_human'])
    enc_m_n.load_state_dict(ckpt_n['encoder_mouse'])
    dec_h_n.load_state_dict(ckpt_n['decoder_human'])
    dec_m_n.load_state_dict(ckpt_n['decoder_mouse'])
    flow_h2m_n.load_state_dict(ckpt_n['flow_h2m'])
    flow_m2h_n = Flow(a['latent_dim'], a['latent_dim']//2, dtype=torch.float).to(device)
    flow_m2h_n.load_state_dict(ckpt_n_m2h['flow_m2h'])

    # Permuted
    enc_h_p, enc_m_p, dec_h_p, dec_m_p, flow_h2m_p = build_models(a, Gh, Gm, device)
    enc_h_p.load_state_dict(ckpt_p['encoder_human'])
    enc_m_p.load_state_dict(ckpt_p['encoder_mouse'])
    dec_h_p.load_state_dict(ckpt_p['decoder_human'])
    dec_m_p.load_state_dict(ckpt_p['decoder_mouse'])
    flow_h2m_p.load_state_dict(ckpt_p['flow_h2m'])
    flow_m2h_p = Flow(a['latent_dim'], a['latent_dim']//2, dtype=torch.float).to(device)
    flow_m2h_p.load_state_dict(ckpt_p_m2h['flow_m2h'])

    for m in [enc_h_n, enc_m_n, dec_h_n, dec_m_n, flow_h2m_n, flow_m2h_n,
              enc_h_p, enc_m_p, dec_h_p, dec_m_p, flow_h2m_p, flow_m2h_p]:
        m.eval()

    return {
        'normal':   {'enc_h': enc_h_n, 'enc_m': enc_m_n, 'dec_h': dec_h_n, 'dec_m': dec_m_n,
                     'flow_h2m': flow_h2m_n, 'flow_m2h': flow_m2h_n},
        'permuted': {'enc_h': enc_h_p, 'enc_m': enc_m_p, 'dec_h': dec_h_p, 'dec_m': dec_m_p,
                     'flow_h2m': flow_h2m_p, 'flow_m2h': flow_m2h_p},
        'args': a, 'Gh': Gh, 'Gm': Gm, 'val_h': val_h, 'val_m': val_m,
    }


# ----------------------------- helpers -----------------------------

def per_sample_pearson(A, B):
    """A, B are (N, G) numpy. Returns (N,) per-sample Pearson r between rows."""
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=1, keepdims=True)
    num = (Ac * Bc).sum(axis=1)
    den = np.sqrt((Ac**2).sum(axis=1) * (Bc**2).sum(axis=1))
    return np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)


def per_gene_pearson(A, B):
    """A, B are (N, G) numpy. Returns (G,) per-gene Pearson r across samples."""
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    num = (Ac * Bc).sum(axis=0)
    den = np.sqrt((Ac**2).sum(axis=0) * (Bc**2).sum(axis=0))
    return np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)


def gene_marginal_mean_var_correlations(A, B):
    """Cross-gene Pearson correlations of sample means and sample variances."""
    mean_a = A.mean(axis=0)
    mean_b = B.mean(axis=0)
    var_a = A.var(axis=0)
    var_b = B.var(axis=0)
    r_mean, _ = pearsonr(mean_a, mean_b)
    r_var, _ = pearsonr(var_a, var_b)
    return float(r_mean), float(r_var)


def pearson_from_sums(sx, sy, sxy, sx2, sy2, n):
    num = n * sxy - sx * sy
    den = np.sqrt(np.maximum(n * sx2 - sx**2, 0) * np.maximum(n * sy2 - sy**2, 0))
    return np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)


def flow_step_n(flow, z, n_steps=10):
    """Run flow ODE for n_steps."""
    device = z.device
    time_steps = torch.linspace(0, 1.0, n_steps + 1, device=device, dtype=torch.float)
    out = z.clone()
    for s in range(n_steps):
        out = flow.step(out, time_steps[s], time_steps[s + 1])
    return out


# ----------------------------- 1. Cycle consistency -----------------------------

def get_cycle_components(comp, cycle_species, mode):
    """Return (enc, flow_forward, flow_back, dec) for the cycle and mode."""
    n, p = comp['normal'], comp['permuted']
    if cycle_species == 'human':
        # x_h â†’ enc_h â†’ flow_h2m â†’ flow_m2h â†’ dec_h
        if mode == 'FlowTransOP':       return n['enc_h'], n['flow_h2m'], n['flow_m2h'], n['dec_h']
        if mode == 'permuted_both':     return p['enc_h'], p['flow_h2m'], p['flow_m2h'], p['dec_h']
        if mode == 'permuted_human':    return p['enc_h'], p['flow_h2m'], n['flow_m2h'], p['dec_h']  # home permuted
        if mode == 'permuted_mouse':    return n['enc_h'], n['flow_h2m'], p['flow_m2h'], n['dec_h']  # through permuted
    elif cycle_species == 'mouse':
        # x_m â†’ enc_m â†’ flow_m2h â†’ flow_h2m â†’ dec_m
        if mode == 'FlowTransOP':       return n['enc_m'], n['flow_m2h'], n['flow_h2m'], n['dec_m']
        if mode == 'permuted_both':     return p['enc_m'], p['flow_m2h'], p['flow_h2m'], p['dec_m']
        if mode == 'permuted_mouse':    return p['enc_m'], p['flow_m2h'], n['flow_h2m'], p['dec_m']  # home permuted
        if mode == 'permuted_human':    return n['enc_m'], n['flow_m2h'], p['flow_h2m'], n['dec_m']  # through permuted
    raise ValueError(f"Bad mode={mode} or cycle_species={cycle_species}")


@torch.no_grad()
def run_cycle(X_lazy, enc, flow_f, flow_b, dec, device, bs=2048):
    """Returns (sample_pearsons (N,), gene_marginal_r_mean, gene_marginal_r_var).

    - sample_pearsons[i]: per-sample Pearson(mu_pred[i], x[i]) across genes.
    - gene_marginal_r_mean: cross-gene Pearson(mean_g(mu_pred), mean_g(x)).
    - gene_marginal_r_var:  cross-gene Pearson(mean_g(var_pred), var_g(x)).
    """
    enc.eval(); dec.eval(); flow_f.eval(); flow_b.eval()
    N, G = X_lazy.shape
    sample_r = np.zeros(N, dtype=np.float64)
    sx = np.zeros(G); sy = np.zeros(G); sy2 = np.zeros(G)
    sxv = np.zeros(G)
    n_total = 0

    for c0 in range(0, N, bs):
        c1 = min(c0 + bs, N)
        X = X_lazy[np.arange(c0, c1), :].to(device)
        z = enc(X)
        z_t = flow_step_n(flow_f, z)
        z_b = flow_step_n(flow_b, z_t)
        mu, var = dec(z_b)

        x_np  = X.cpu().numpy()
        mu_np = mu.cpu().numpy()
        v_np  = var.cpu().numpy()

        sample_r[c0:c1] = per_sample_pearson(mu_np, x_np)

        sx  += mu_np.sum(axis=0)
        sy  += x_np.sum(axis=0)
        sy2 += (x_np**2).sum(axis=0)

        sxv += v_np.sum(axis=0)

        n_total += (c1 - c0)

    # mu marginal: mean of predicted mu per gene  vs  mean of true x per gene
    mean_mu_per_gene = sx / n_total
    mean_x_per_gene  = sy / n_total
    r_mu_marginal, _ = pearsonr(mean_mu_per_gene, mean_x_per_gene)

    # var marginal: mean of predicted variance per gene  vs  actual sample variance of x per gene
    mean_var_per_gene = sxv / n_total
    var_x_per_gene    = (sy2 / n_total) - mean_x_per_gene**2
    r_var_marginal, _ = pearsonr(mean_var_per_gene, var_x_per_gene)

    return sample_r, float(r_mu_marginal), float(r_var_marginal)


def cycle_evaluation(comp, fold, device, log):
    modes = ['FlowTransOP', 'permuted_both', 'permuted_human', 'permuted_mouse']

    for cycle_species, val_lazy in [
        ('human', comp['val_h']),
        ('mouse', comp['val_m']),
    ]:
        log(f"\n=== Cycle consistency: {cycle_species} ===")
        rows_ps = []

        for mode in modes:
            log(f"  Mode: {mode}")
            enc, flow_f, flow_b, dec = get_cycle_components(comp, cycle_species, mode)
            sample_r, r_mean_marg, r_var_marg = run_cycle(
                val_lazy, enc, flow_f, flow_b, dec, device)
            log(f"    per-sample <r> = {sample_r.mean():.4f} | "
                f"gene-marginal mean r = {r_mean_marg:.4f} | "
                f"gene-marginal variance r = {r_var_marg:.4f}")

            rows_ps.append({
                'fold': fold,
                'per_sample_mean':       float(sample_r.mean()),
                'gene_marginal_r_mean':  r_mean_marg,
                'gene_marginal_r_var':   r_var_marg,
                'model_type': mode,
            })

        pd.DataFrame(rows_ps).to_csv(EVAL_DIR / f"cycle_{cycle_species}_persample_fold{fold}.csv", index=False)
        stale = EVAL_DIR / f"cycle_{cycle_species}_pergene_fold{fold}.csv"
        if stale.exists():
            stale.unlink()


# ----------------------------- 2. Orthologue-mediated -----------------------------

def build_orthologue_map(human_genes, mouse_genes):
    """Case-insensitive symbol match. Catches the bulk of canonical orthologues
    (ABCD1â†”Abcd1, GAPDHâ†”Gapdh) but misses renamed pairs (TP53â†”Trp53). Replace
    with a real HGNCâ†”MGI orthologue map for paper-grade analysis."""
    h_lower = {str(g).lower(): i for i, g in enumerate(human_genes)}
    m_lower = {str(g).lower(): i for i, g in enumerate(mouse_genes)}
    common = sorted(set(h_lower) & set(m_lower))
    h_idx = np.array([h_lower[c] for c in common], dtype=np.int64)
    m_idx = np.array([m_lower[c] for c in common], dtype=np.int64)
    return h_idx, m_idx


@torch.no_grad()
def translate_full(X_lazy, enc, flow, dec, device, out_dim, bs=2048):
    """Run enc â†’ flow â†’ dec, return (N, out_dim) numpy mu predictions."""
    enc.eval(); flow.eval(); dec.eval()
    N = len(X_lazy)
    out = np.empty((N, out_dim), dtype=np.float32)
    for c0 in range(0, N, bs):
        c1 = min(c0 + bs, N)
        X = X_lazy[np.arange(c0, c1), :].to(device)
        z = enc(X)
        z_t = flow_step_n(flow, z)
        mu, _ = dec(z_t)
        out[c0:c1] = mu.cpu().numpy()
    return out


def gather_orthologue(X_lazy, ortho_idx, bs=2048):
    """Return (N, n_ortho) numpy of X subset to orthologue gene indices."""
    N = len(X_lazy)
    out = np.empty((N, len(ortho_idx)), dtype=np.float32)
    for c0 in range(0, N, bs):
        c1 = min(c0 + bs, N)
        out[c0:c1] = X_lazy[np.arange(c0, c1), :].numpy()[:, ortho_idx]
    return out


def orthologue_eval(comp, fold, device, log):
    log("\n=== Orthologue-mediated comparison ===")
    h_genes = load_genes("human")
    m_genes = load_genes("mouse")
    h_idx, m_idx = build_orthologue_map(h_genes, m_genes)
    log(f"  {len(h_idx)} orthologues by case-insensitive symbol match")

    n, p = comp['normal'], comp['permuted']

    # 4 modes per leg, mixing encoder/flow/decoder from normal vs permuted
    modes_h2m = {
        'FlowTransOP':    (n['enc_h'], n['flow_h2m'], n['dec_m']),
        'permuted_both':  (p['enc_h'], p['flow_h2m'], p['dec_m']),
        'permuted_human': (p['enc_h'], p['flow_h2m'], n['dec_m']),  # source side broken
        'permuted_mouse': (n['enc_h'], p['flow_h2m'], p['dec_m']),  # target side broken
    }
    modes_m2h = {
        'FlowTransOP':    (n['enc_m'], n['flow_m2h'], n['dec_h']),
        'permuted_both':  (p['enc_m'], p['flow_m2h'], p['dec_h']),
        'permuted_mouse': (p['enc_m'], p['flow_m2h'], n['dec_h']),  # source side broken
        'permuted_human': (n['enc_m'], p['flow_m2h'], p['dec_h']),  # target side broken
    }

    # Pre-gather original val data restricted to orthologue genes (used as comparison vector)
    log("  gathering source-side orthologue values...")
    x_h_orth = gather_orthologue(comp['val_h'], h_idx)
    x_m_orth = gather_orthologue(comp['val_m'], m_idx)

    rows_h2m = []
    rows_h2m_summary = []
    for mode, (enc, flow, dec) in modes_h2m.items():
        log(f"  h2m mode: {mode}")
        mu_pred = translate_full(comp['val_h'], enc, flow, dec, device, comp['Gm'])
        mu_pred_orth = mu_pred[:, m_idx]
        rs = per_sample_pearson(mu_pred_orth, x_h_orth)
        r_mean_marg, r_var_marg = gene_marginal_mean_var_correlations(mu_pred_orth, x_h_orth)
        for i, r in enumerate(rs):
            rows_h2m.append({'fold': fold, 'direction': 'h2m', 'sample_idx': i,
                             'pearson': float(r), 'model_type': mode})
        rows_h2m_summary.append({
            'fold': fold,
            'direction': 'h2m',
            'n_samples': int(mu_pred_orth.shape[0]),
            'n_orthologues': int(mu_pred_orth.shape[1]),
            'per_sample_mean': float(rs.mean()),
            'gene_marginal_r_mean': r_mean_marg,
            'gene_marginal_r_var': r_var_marg,
            'model_type': mode,
        })
        del mu_pred, mu_pred_orth
        log(f"    sample mean r(source orthologues) = {rs.mean():.4f} | "
            f"gene-marginal mean r = {r_mean_marg:.4f} | "
            f"gene-marginal variance r = {r_var_marg:.4f}")

    rows_m2h = []
    rows_m2h_summary = []
    for mode, (enc, flow, dec) in modes_m2h.items():
        log(f"  m2h mode: {mode}")
        mu_pred = translate_full(comp['val_m'], enc, flow, dec, device, comp['Gh'])
        mu_pred_orth = mu_pred[:, h_idx]
        rs = per_sample_pearson(mu_pred_orth, x_m_orth)
        r_mean_marg, r_var_marg = gene_marginal_mean_var_correlations(mu_pred_orth, x_m_orth)
        for i, r in enumerate(rs):
            rows_m2h.append({'fold': fold, 'direction': 'm2h', 'sample_idx': i,
                             'pearson': float(r), 'model_type': mode})
        rows_m2h_summary.append({
            'fold': fold,
            'direction': 'm2h',
            'n_samples': int(mu_pred_orth.shape[0]),
            'n_orthologues': int(mu_pred_orth.shape[1]),
            'per_sample_mean': float(rs.mean()),
            'gene_marginal_r_mean': r_mean_marg,
            'gene_marginal_r_var': r_var_marg,
            'model_type': mode,
        })
        del mu_pred, mu_pred_orth
        log(f"    sample mean r(source orthologues) = {rs.mean():.4f} | "
            f"gene-marginal mean r = {r_mean_marg:.4f} | "
            f"gene-marginal variance r = {r_var_marg:.4f}")

    pd.DataFrame(rows_h2m).to_csv(EVAL_DIR / f"orthologue_h2m_fold{fold}.csv", index=False)
    pd.DataFrame(rows_m2h).to_csv(EVAL_DIR / f"orthologue_m2h_fold{fold}.csv", index=False)
    pd.DataFrame(rows_h2m_summary).to_csv(EVAL_DIR / f"orthologue_h2m_summary_fold{fold}.csv", index=False)
    pd.DataFrame(rows_m2h_summary).to_csv(EVAL_DIR / f"orthologue_m2h_summary_fold{fold}.csv", index=False)
    for stale in [
        EVAL_DIR / f"orthologue_h2m_pergene_fold{fold}.csv",
        EVAL_DIR / f"orthologue_m2h_pergene_fold{fold}.csv",
    ]:
        if stale.exists():
            stale.unlink()


# ----------------------------- 3. MMD -----------------------------

def mmd_rbf(X, Y, max_n=5000, seed=0, sigma=None):
    """Biased MMDÂ² with RBF kernel and median heuristic; subsamples to max_n."""
    rng = np.random.default_rng(seed)
    n = min(max_n, X.shape[0]); m = min(max_n, Y.shape[0])
    Xs = X[rng.choice(X.shape[0], n, replace=False)]
    Ys = Y[rng.choice(Y.shape[0], m, replace=False)]
    Xs = torch.as_tensor(Xs, dtype=torch.float32)
    Ys = torch.as_tensor(Ys, dtype=torch.float32)
    XX = torch.cdist(Xs, Xs).pow(2)
    YY = torch.cdist(Ys, Ys).pow(2)
    XY = torch.cdist(Xs, Ys).pow(2)
    if sigma is None:
        d2 = torch.cat([XX.flatten(), YY.flatten(), XY.flatten()])
        sigma = d2.median().sqrt().item() + 1e-8
    g = 1.0 / (2 * sigma**2)
    return float((torch.exp(-g*XX).mean()
                  + torch.exp(-g*YY).mean()
                  - 2*torch.exp(-g*XY).mean()).item())


def mmd_eval(fold, log):
    log("\n=== MMD on latents ===")
    z_h_val        = np.load(MODEL_DIR / f"fold_{fold}_z_human_val.npy")
    z_m_val        = np.load(MODEL_DIR / f"fold_{fold}_z_mouse_val.npy")
    z_h2m_val      = np.load(MODEL_DIR / f"fold_{fold}_z_h2m_val.npy")
    z_m2h_val      = np.load(MODEL_DIR / f"fold_{fold}_z_m2h_val.npy")
    z_h_val_perm   = np.load(MODEL_DIR / f"fold_{fold}_z_human_val_perm.npy")
    z_m_val_perm   = np.load(MODEL_DIR / f"fold_{fold}_z_mouse_val_perm.npy")
    z_h2m_val_perm = np.load(MODEL_DIR / f"fold_{fold}_z_h2m_val_perm.npy")
    z_m2h_val_perm = np.load(MODEL_DIR / f"fold_{fold}_z_m2h_val_perm.npy")

    rows = []
    # h2m: source (real human) vs target (real mouse), no translation = upper bound
    rows.append({'fold': fold, 'direction': 'h2m', 'model_type': 'no_translation',
                 'mmd': mmd_rbf(z_h_val, z_m_val)})
    rows.append({'fold': fold, 'direction': 'h2m', 'model_type': 'FlowTransOP',
                 'mmd': mmd_rbf(z_h2m_val, z_m_val)})
    # Permuted is in its own latent space, so compare within that space
    rows.append({'fold': fold, 'direction': 'h2m', 'model_type': 'permuted_both',
                 'mmd': mmd_rbf(z_h2m_val_perm, z_m_val_perm)})

    rows.append({'fold': fold, 'direction': 'm2h', 'model_type': 'no_translation',
                 'mmd': mmd_rbf(z_m_val, z_h_val)})
    rows.append({'fold': fold, 'direction': 'm2h', 'model_type': 'FlowTransOP',
                 'mmd': mmd_rbf(z_m2h_val, z_h_val)})
    rows.append({'fold': fold, 'direction': 'm2h', 'model_type': 'permuted_both',
                 'mmd': mmd_rbf(z_m2h_val_perm, z_h_val_perm)})

    pd.DataFrame(rows).to_csv(EVAL_DIR / f"mmd_fold{fold}.csv", index=False)
    for r in rows:
        log(f"  {r['direction']:>3s} {r['model_type']:>20s}: MMDÂ² = {r['mmd']:.6f}")


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--skip_cycle", action='store_true')
    parser.add_argument("--skip_orth",  action='store_true')
    parser.add_argument("--skip_mmd",   action='store_true')
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    log_file = f'logs/evaluate_translation_fold{args.fold}.log'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = FileHandler(log_file, mode='a'); fh.setFormatter(logging.Formatter('%(message)s'))
    sh = logging.StreamHandler();          sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh); logger.addHandler(sh)
    log = logger.info

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Fold {args.fold} | Device: {device}")

    log("Loading all componentsâ€¦")
    comp = load_all_components(args.fold, device)
    log(f"  Gh={comp['Gh']}, Gm={comp['Gm']}, latent_dim={comp['args']['latent_dim']}")

    if not args.skip_cycle: cycle_evaluation(comp, args.fold, device, log)
    if not args.skip_orth:  orthologue_eval(comp, args.fold, device, log)
    if not args.skip_mmd:   mmd_eval(args.fold, log)

    log(f"\nâœ“ Fold {args.fold} evaluation complete.")


if __name__ == "__main__":
    main()
