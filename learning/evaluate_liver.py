#!/usr/bin/env python3
"""
Combined FlowTransOP evaluation on the external ARCHS4 liver test set.

This script reuses the trained fold models but replaces the validation matrices
with the liver-only external test matrices written by preprocess_archs4.py:

  ../archs4/preprocessed/human_test_X.npy
  ../archs4/preprocessed/mouse_test_X.npy

It combines the evaluation families from:
  - evaluate_translation.py: cycle consistency, orthologue translation, latent MMD
  - evaluate_tissue.py: liver-specific centroid/neighborhood diagnostics
  - evaluate_expression_mmd_archs4.py: expression-space MMD

Because the external test set is intentionally liver-only, multi-class tissue
metrics such as kNN macro-F1, linear probing, variance partitioning by tissue,
and tissue-rank retrieval are not identifiable here. Instead, this script writes
liver-specific centroid similarity and nearest-target similarity diagnostics.
"""

import argparse
import logging
from logging import FileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

import evaluate_translation as et


DATA_DIR = Path("../archs4")
PREPROC_DIR = DATA_DIR / "preprocessed"
EVAL_DIR = DATA_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["FlowTransOP", "permuted_both", "permuted_human", "permuted_mouse"]


def setup_logger(fold):
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger("evaluate_liver")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    fh = FileHandler(f"logs/evaluate_liver_fold{fold}.log", mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger.info


def load_liver_components(fold, device):
    comp = et.load_all_components(fold, device)
    comp["val_h"] = et.LazyMatrix(PREPROC_DIR / "human_test_X.npy")
    comp["val_m"] = et.LazyMatrix(PREPROC_DIR / "mouse_test_X.npy")
    comp["train_h"] = et.LazyMatrix(PREPROC_DIR / "human_X.npy")
    comp["train_m"] = et.LazyMatrix(PREPROC_DIR / "mouse_X.npy")
    comp["Gh"] = comp["val_h"].shape[1]
    comp["Gm"] = comp["val_m"].shape[1]
    return comp


def sample_indices(n, max_n, rng):
    if max_n is None or max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=int(max_n), replace=False))


def safe_corr(a, b, method="pearson"):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    if method == "spearman":
        return float(spearmanr(a, b).correlation)
    return float(pearsonr(a, b)[0])


def safe_cosine(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-12:
        return np.nan
    return float(np.dot(a, b) / den)


def gather_rows(X_lazy, rows, cols=None, bs=512):
    cols = None if cols is None else np.asarray(cols, dtype=np.int64)
    n_features = X_lazy.shape[1] if cols is None else len(cols)
    out = np.empty((len(rows), n_features), dtype=np.float32)
    for start in range(0, len(rows), bs):
        stop = min(start + bs, len(rows))
        block = X_lazy[rows[start:stop], :].numpy()
        if cols is not None:
            block = block[:, cols]
        out[start:stop] = block
    return out


@torch.no_grad()
def translate_rows(X_lazy, rows, enc, flow, dec, device, out_dim, cols=None, bs=256):
    enc.eval()
    flow.eval()
    dec.eval()
    cols = None if cols is None else np.asarray(cols, dtype=np.int64)
    n_features = out_dim if cols is None else len(cols)
    out = np.empty((len(rows), n_features), dtype=np.float32)
    for start in range(0, len(rows), bs):
        stop = min(start + bs, len(rows))
        X = X_lazy[rows[start:stop], :].to(device)
        z = enc(X)
        z_t = et.flow_step_n(flow, z)
        mu, _ = dec(z_t)
        block = mu.cpu().numpy()
        if cols is not None:
            block = block[:, cols]
        out[start:stop] = block
    return out


@torch.no_grad()
def latent_rows(X_lazy, rows, enc, device, flow=None, bs=512):
    enc.eval()
    if flow is not None:
        flow.eval()
    blocks = []
    for start in range(0, len(rows), bs):
        stop = min(start + bs, len(rows))
        X = X_lazy[rows[start:stop], :].to(device)
        z = enc(X)
        if flow is not None:
            z = et.flow_step_n(flow, z)
        blocks.append(z.cpu().numpy())
    return np.vstack(blocks).astype(np.float32)


def translation_modes(comp, direction):
    n, p = comp["normal"], comp["permuted"]
    if direction == "h2m":
        return {
            "FlowTransOP": (n["enc_h"], n["flow_h2m"], n["dec_m"]),
            "permuted_both": (p["enc_h"], p["flow_h2m"], p["dec_m"]),
            "permuted_human": (p["enc_h"], p["flow_h2m"], n["dec_m"]),
            "permuted_mouse": (n["enc_h"], n["flow_h2m"], p["dec_m"]),
        }
    if direction == "m2h":
        return {
            "FlowTransOP": (n["enc_m"], n["flow_m2h"], n["dec_h"]),
            "permuted_both": (p["enc_m"], p["flow_m2h"], p["dec_h"]),
            "permuted_mouse": (p["enc_m"], p["flow_m2h"], n["dec_h"]),
            "permuted_human": (n["enc_m"], n["flow_m2h"], p["dec_h"]),
        }
    raise ValueError(f"Unknown direction: {direction}")


def centroid_row(pred, target, fold, direction, model_type, space, feature_set, comparison):
    pred_cent = pred.mean(axis=0)
    target_cent = target.mean(axis=0)
    return {
        "fold": fold,
        "direction": direction,
        "model_type": model_type,
        "space": space,
        "feature_set": feature_set,
        "comparison": comparison,
        "n_source": pred.shape[0],
        "n_target": target.shape[0],
        "n_features": pred.shape[1],
        "centroid_euclidean": float(np.linalg.norm(pred_cent - target_cent)),
        "centroid_cosine": safe_cosine(pred_cent, target_cent),
        "centroid_pearson": safe_corr(pred_cent, target_cent, "pearson"),
        "centroid_spearman": safe_corr(pred_cent, target_cent, "spearman"),
    }


def nearest_target_summary(pred, target, k=10, chunk_size=128):
    kk = min(int(k), target.shape[0])
    if kk < 1:
        return {"mean_top1_cosine": np.nan, "mean_topk_cosine": np.nan}
    pred_norm = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    target_norm = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    top1, topk = [], []
    for start in range(0, pred_norm.shape[0], chunk_size):
        stop = min(start + chunk_size, pred_norm.shape[0])
        sim = pred_norm[start:stop] @ target_norm.T
        top_idx = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
        top_sim = np.take_along_axis(sim, top_idx, axis=1)
        top1.extend(top_sim.max(axis=1).tolist())
        topk.extend(top_sim.mean(axis=1).tolist())
    return {
        "nearest_k": kk,
        "mean_top1_cosine": float(np.mean(top1)),
        "mean_topk_cosine": float(np.mean(topk)),
    }


def mmd_value(X, Y, max_n, seed):
    cap = max(X.shape[0], Y.shape[0]) if max_n is None or max_n <= 0 else max_n
    return et.mmd_rbf(X, Y, max_n=cap, seed=seed)


def background_rows(X_lazy, n_samples, rng):
    take = min(int(n_samples), len(X_lazy))
    return np.sort(rng.choice(len(X_lazy), size=take, replace=False))


def cycle_liver(comp, fold, device, bs, log):
    log("\n=== Liver cycle consistency ===")
    for species, X_lazy in [("human", comp["val_h"]), ("mouse", comp["val_m"])]:
        rows_ps = []
        log(f"  {species}: {len(X_lazy)} liver samples")
        for mode in MODES:
            enc, flow_f, flow_b, dec = et.get_cycle_components(comp, species, mode)
            sample_r, r_mean_marg, r_var_marg = et.run_cycle(
                X_lazy, enc, flow_f, flow_b, dec, device, bs=bs
            )
            rows_ps.append({
                "fold": fold,
                "species": species,
                "tissue": "liver",
                "n_samples": len(X_lazy),
                "per_sample_mean": float(sample_r.mean()),
                "gene_marginal_r_mean": r_mean_marg,
                "gene_marginal_r_var": r_var_marg,
                "model_type": mode,
            })
            log(f"    {mode}: sample r={sample_r.mean():.4f}, "
                f"gene marginal mean r={r_mean_marg:.4f}, "
                f"gene marginal variance r={r_var_marg:.4f}")

        pd.DataFrame(rows_ps).to_csv(
            EVAL_DIR / f"liver_cycle_{species}_persample_fold{fold}.csv", index=False
        )
        stale = EVAL_DIR / f"liver_cycle_{species}_pergene_fold{fold}.csv"
        if stale.exists():
            stale.unlink()


def orthologue_liver(comp, fold, device, bs, log):
    log("\n=== Liver orthologue-mediated translation ===")
    h_genes = et.load_genes("human")
    m_genes = et.load_genes("mouse")
    h_idx, m_idx = et.build_orthologue_map(h_genes, m_genes)
    log(f"  matched orthologues: {len(h_idx)}")

    h_rows = np.arange(len(comp["val_h"]), dtype=np.int64)
    m_rows = np.arange(len(comp["val_m"]), dtype=np.int64)
    x_h_orth = gather_rows(comp["val_h"], h_rows, h_idx, bs=bs)
    x_m_orth = gather_rows(comp["val_m"], m_rows, m_idx, bs=bs)

    for direction in ["h2m", "m2h"]:
        rows_ps = []
        rows_summary = []
        if direction == "h2m":
            source_X, source_rows = comp["val_h"], h_rows
            source_orth = x_h_orth
            target_cols, out_dim = m_idx, comp["Gm"]
        else:
            source_X, source_rows = comp["val_m"], m_rows
            source_orth = x_m_orth
            target_cols, out_dim = h_idx, comp["Gh"]

        for mode, (enc, flow, dec) in translation_modes(comp, direction).items():
            pred_orth = translate_rows(
                source_X, source_rows, enc, flow, dec, device, out_dim,
                cols=target_cols, bs=bs
            )
            rs = et.per_sample_pearson(pred_orth, source_orth)
            r_mean_marg, r_var_marg = et.gene_marginal_mean_var_correlations(pred_orth, source_orth)
            for i, r in enumerate(rs):
                rows_ps.append({
                    "fold": fold,
                    "direction": direction,
                    "tissue": "liver",
                    "sample_idx": i,
                    "pearson": float(r),
                    "model_type": mode,
                })
            rows_summary.append({
                "fold": fold,
                "direction": direction,
                "tissue": "liver",
                "n_samples": int(pred_orth.shape[0]),
                "n_orthologues": int(pred_orth.shape[1]),
                "per_sample_mean": float(rs.mean()),
                "gene_marginal_r_mean": r_mean_marg,
                "gene_marginal_r_var": r_var_marg,
                "model_type": mode,
            })
            log(f"  {direction} {mode}: source-orth sample r={rs.mean():.4f}, "
                f"gene marginal mean r={r_mean_marg:.4f}, "
                f"gene marginal variance r={r_var_marg:.4f}")

        pd.DataFrame(rows_ps).to_csv(
            EVAL_DIR / f"liver_orthologue_{direction}_fold{fold}.csv", index=False
        )
        pd.DataFrame(rows_summary).to_csv(
            EVAL_DIR / f"liver_orthologue_{direction}_summary_fold{fold}.csv", index=False
        )
        for stale in [
            EVAL_DIR / f"liver_orthologue_{direction}_pergene_fold{fold}.csv",
        ]:
            if stale.exists():
                stale.unlink()


def latent_mmd_liver(comp, fold, device, max_n, bs, seed, log,
                     background_repeats=10, background_samples=1000):
    log("\n=== Liver latent MMD ===")
    rng = np.random.default_rng(seed + fold)
    h_rows = sample_indices(len(comp["val_h"]), max_n, rng)
    m_rows = sample_indices(len(comp["val_m"]), max_n, rng)
    n, p = comp["normal"], comp["permuted"]

    z_h = latent_rows(comp["val_h"], h_rows, n["enc_h"], device, bs=bs)
    z_m = latent_rows(comp["val_m"], m_rows, n["enc_m"], device, bs=bs)
    z_h2m = latent_rows(comp["val_h"], h_rows, n["enc_h"], device, flow=n["flow_h2m"], bs=bs)
    z_m2h = latent_rows(comp["val_m"], m_rows, n["enc_m"], device, flow=n["flow_m2h"], bs=bs)

    z_h_p = latent_rows(comp["val_h"], h_rows, p["enc_h"], device, bs=bs)
    z_m_p = latent_rows(comp["val_m"], m_rows, p["enc_m"], device, bs=bs)
    z_h2m_p = latent_rows(comp["val_h"], h_rows, p["enc_h"], device, flow=p["flow_h2m"], bs=bs)
    z_m2h_p = latent_rows(comp["val_m"], m_rows, p["enc_m"], device, flow=p["flow_m2h"], bs=bs)

    rows = [
        {
            "fold": fold, "direction": "h2m", "tissue": "liver",
            "model_type": "FlowTransOP", "comparison": "translated_source_vs_real_target",
            "n_source": len(h_rows), "n_target": len(m_rows), "n_features": z_m.shape[1],
            "mmd": mmd_value(z_h2m, z_m, max_n=max_n, seed=seed + fold),
        },
        {
            "fold": fold, "direction": "h2m", "tissue": "liver",
            "model_type": "permuted_both", "comparison": "translated_source_vs_real_target",
            "n_source": len(h_rows), "n_target": len(m_rows), "n_features": z_m_p.shape[1],
            "mmd": mmd_value(z_h2m_p, z_m_p, max_n=max_n, seed=seed + fold),
        },
        {
            "fold": fold, "direction": "m2h", "tissue": "liver",
            "model_type": "FlowTransOP", "comparison": "translated_source_vs_real_target",
            "n_source": len(m_rows), "n_target": len(h_rows), "n_features": z_h.shape[1],
            "mmd": mmd_value(z_m2h, z_h, max_n=max_n, seed=seed + fold),
        },
        {
            "fold": fold, "direction": "m2h", "tissue": "liver",
            "model_type": "permuted_both", "comparison": "translated_source_vs_real_target",
            "n_source": len(m_rows), "n_target": len(h_rows), "n_features": z_h_p.shape[1],
            "mmd": mmd_value(z_m2h_p, z_h_p, max_n=max_n, seed=seed + fold),
        },
    ]
    pd.DataFrame(rows).to_csv(EVAL_DIR / f"liver_latent_mmd_fold{fold}.csv", index=False)

    cent_rows = [
        centroid_row(z_h2m, z_m, fold, "h2m", "FlowTransOP", "latent", "latent", "translated_source_vs_real_target"),
        centroid_row(z_h2m_p, z_m_p, fold, "h2m", "permuted_both", "latent", "latent", "translated_source_vs_real_target"),
        centroid_row(z_m2h, z_h, fold, "m2h", "FlowTransOP", "latent", "latent", "translated_source_vs_real_target"),
        centroid_row(z_m2h_p, z_h_p, fold, "m2h", "permuted_both", "latent", "latent", "translated_source_vs_real_target"),
    ]

    for repeat in range(background_repeats):
        bg_m_rows = background_rows(comp["train_m"], background_samples, rng)
        bg_h_rows = background_rows(comp["train_h"], background_samples, rng)
        z_m_bg = latent_rows(comp["train_m"], bg_m_rows, n["enc_m"], device, bs=bs)
        z_h_bg = latent_rows(comp["train_h"], bg_h_rows, n["enc_h"], device, bs=bs)
        z_m_bg_p = latent_rows(comp["train_m"], bg_m_rows, p["enc_m"], device, bs=bs)
        z_h_bg_p = latent_rows(comp["train_h"], bg_h_rows, p["enc_h"], device, bs=bs)

        background_specs = [
            ("h2m", "target_liver", z_m, z_m_bg, "normal_target_space"),
            ("h2m", "FlowTransOP", z_h2m, z_m_bg, "normal_target_space"),
            ("h2m", "target_liver_permuted_space", z_m_p, z_m_bg_p, "permuted_target_space"),
            ("h2m", "permuted_both", z_h2m_p, z_m_bg_p, "permuted_target_space"),
            ("m2h", "target_liver", z_h, z_h_bg, "normal_target_space"),
            ("m2h", "FlowTransOP", z_m2h, z_h_bg, "normal_target_space"),
            ("m2h", "target_liver_permuted_space", z_h_p, z_h_bg_p, "permuted_target_space"),
            ("m2h", "permuted_both", z_m2h_p, z_h_bg_p, "permuted_target_space"),
        ]
        for direction, model_type, liver_latent, background_latent, latent_space in background_specs:
            comparison = (
                "real_target_liver_vs_random_non_liver_target"
                if model_type.startswith("target_liver")
                else "translated_liver_source_vs_random_non_liver_target"
            )
            row = centroid_row(
                liver_latent, background_latent, fold, direction, model_type,
                "latent", "latent", comparison
            )
            row.update({
                "random_repeat": repeat,
                "background_n": int(background_latent.shape[0]),
                "latent_space": latent_space,
            })
            cent_rows.append(row)

    pd.DataFrame(cent_rows).to_csv(
        EVAL_DIR / f"liver_centroid_latent_fold{fold}.csv", index=False
    )
    for row in rows:
        log(f"  {row['direction']} {row['model_type']}: MMD={row['mmd']:.6f}")


def expression_mmd_liver(comp, fold, device, max_n, bs, seed, log,
                         background_repeats=10, background_samples=1000):
    log("\n=== Liver expression-space MMD ===")
    h_genes = et.load_genes("human")
    m_genes = et.load_genes("mouse")
    h_idx, m_idx = et.build_orthologue_map(h_genes, m_genes)

    rng = np.random.default_rng(seed + fold)
    h_rows = sample_indices(len(comp["val_h"]), max_n, rng)
    m_rows = sample_indices(len(comp["val_m"]), max_n, rng)

    x_h_orth = gather_rows(comp["val_h"], h_rows, h_idx, bs=bs)
    x_m_orth = gather_rows(comp["val_m"], m_rows, m_idx, bs=bs)
    baseline_mmd = mmd_value(x_h_orth, x_m_orth, max_n=max_n, seed=seed + fold)

    rows = []
    for direction in ["h2m", "m2h"]:
        for feature_set, comparison in [
            ("orthologues", "source_liver_test_vs_target_liver_test"),
            ("all_target_genes", "source_liver_test_vs_target_liver_test_orthologue_reference"),
        ]:
            rows.append({
                "fold": fold,
                "direction": direction,
                "tissue": "liver",
                "feature_set": feature_set,
                "model_type": "liver_test_orthologues",
                "comparison": comparison,
                "n_source": len(h_rows) if direction == "h2m" else len(m_rows),
                "n_target": len(m_rows) if direction == "h2m" else len(h_rows),
                "n_features": len(h_idx),
                "mmd": baseline_mmd,
            })

    target_mouse_all = gather_rows(comp["val_m"], m_rows, cols=None, bs=bs)
    target_mouse_orth = target_mouse_all[:, m_idx]
    for mode, (enc, flow, dec) in translation_modes(comp, "h2m").items():
        log(f"  h2m {mode}: translating {len(h_rows)} liver samples")
        pred_all = translate_rows(comp["val_h"], h_rows, enc, flow, dec, device, comp["Gm"], bs=bs)
        pred_orth = pred_all[:, m_idx]
        rows.append({
            "fold": fold, "direction": "h2m", "tissue": "liver",
            "feature_set": "all_target_genes", "model_type": mode,
            "comparison": "translated_liver_source_vs_liver_target",
            "n_source": len(h_rows), "n_target": len(m_rows), "n_features": comp["Gm"],
            "mmd": mmd_value(pred_all, target_mouse_all, max_n=max_n, seed=seed + fold),
        })
        rows.append({
            "fold": fold, "direction": "h2m", "tissue": "liver",
            "feature_set": "orthologues", "model_type": mode,
            "comparison": "translated_liver_source_vs_liver_target",
            "n_source": len(h_rows), "n_target": len(m_rows), "n_features": len(m_idx),
            "mmd": mmd_value(pred_orth, target_mouse_orth, max_n=max_n, seed=seed + fold),
        })
        for repeat in range(background_repeats):
            bg_rows = background_rows(comp["train_m"], background_samples, rng)
            bg_all = gather_rows(comp["train_m"], bg_rows, cols=None, bs=bs)
            bg_orth = bg_all[:, m_idx]
            for feature_set, pred, target_liver, bg in [
                ("all_target_genes", pred_all, target_mouse_all, bg_all),
                ("orthologues", pred_orth, target_mouse_orth, bg_orth),
            ]:
                if mode == "FlowTransOP":
                    rows.append({
                        "fold": fold, "direction": "h2m", "tissue": "liver",
                        "feature_set": feature_set, "model_type": "target_liver",
                        "comparison": "real_target_liver_vs_random_non_liver_target",
                        "random_repeat": repeat,
                        "n_source": target_liver.shape[0], "n_target": bg.shape[0],
                        "n_features": bg.shape[1],
                        "mmd": mmd_value(target_liver, bg, max_n=max_n, seed=seed + fold + repeat),
                    })
                rows.append({
                    "fold": fold, "direction": "h2m", "tissue": "liver",
                    "feature_set": feature_set, "model_type": mode,
                    "comparison": "translated_liver_source_vs_random_non_liver_target",
                    "random_repeat": repeat,
                    "n_source": pred.shape[0], "n_target": bg.shape[0],
                    "n_features": bg.shape[1],
                    "mmd": mmd_value(pred, bg, max_n=max_n, seed=seed + fold + repeat),
                })
        del pred_all, pred_orth

    target_human_all = gather_rows(comp["val_h"], h_rows, cols=None, bs=bs)
    target_human_orth = target_human_all[:, h_idx]
    for mode, (enc, flow, dec) in translation_modes(comp, "m2h").items():
        log(f"  m2h {mode}: translating {len(m_rows)} liver samples")
        pred_all = translate_rows(comp["val_m"], m_rows, enc, flow, dec, device, comp["Gh"], bs=bs)
        pred_orth = pred_all[:, h_idx]
        rows.append({
            "fold": fold, "direction": "m2h", "tissue": "liver",
            "feature_set": "all_target_genes", "model_type": mode,
            "comparison": "translated_liver_source_vs_liver_target",
            "n_source": len(m_rows), "n_target": len(h_rows), "n_features": comp["Gh"],
            "mmd": mmd_value(pred_all, target_human_all, max_n=max_n, seed=seed + fold),
        })
        rows.append({
            "fold": fold, "direction": "m2h", "tissue": "liver",
            "feature_set": "orthologues", "model_type": mode,
            "comparison": "translated_liver_source_vs_liver_target",
            "n_source": len(m_rows), "n_target": len(h_rows), "n_features": len(h_idx),
            "mmd": mmd_value(pred_orth, target_human_orth, max_n=max_n, seed=seed + fold),
        })
        for repeat in range(background_repeats):
            bg_rows = background_rows(comp["train_h"], background_samples, rng)
            bg_all = gather_rows(comp["train_h"], bg_rows, cols=None, bs=bs)
            bg_orth = bg_all[:, h_idx]
            for feature_set, pred, target_liver, bg in [
                ("all_target_genes", pred_all, target_human_all, bg_all),
                ("orthologues", pred_orth, target_human_orth, bg_orth),
            ]:
                if mode == "FlowTransOP":
                    rows.append({
                        "fold": fold, "direction": "m2h", "tissue": "liver",
                        "feature_set": feature_set, "model_type": "target_liver",
                        "comparison": "real_target_liver_vs_random_non_liver_target",
                        "random_repeat": repeat,
                        "n_source": target_liver.shape[0], "n_target": bg.shape[0],
                        "n_features": bg.shape[1],
                        "mmd": mmd_value(target_liver, bg, max_n=max_n, seed=seed + fold + repeat),
                    })
                rows.append({
                    "fold": fold, "direction": "m2h", "tissue": "liver",
                    "feature_set": feature_set, "model_type": mode,
                    "comparison": "translated_liver_source_vs_random_non_liver_target",
                    "random_repeat": repeat,
                    "n_source": pred.shape[0], "n_target": bg.shape[0],
                    "n_features": bg.shape[1],
                    "mmd": mmd_value(pred, bg, max_n=max_n, seed=seed + fold + repeat),
                })
        del pred_all, pred_orth

    pd.DataFrame(rows).to_csv(EVAL_DIR / f"liver_expression_mmd_fold{fold}.csv", index=False)
    log(f"  baseline liver orthologue MMD={baseline_mmd:.6f}")


def expression_centroid_liver(comp, fold, device, max_n, bs, seed, k, log):
    log("\n=== Liver expression centroid/neighborhood diagnostics ===")
    h_genes = et.load_genes("human")
    m_genes = et.load_genes("mouse")
    h_idx, m_idx = et.build_orthologue_map(h_genes, m_genes)

    rng = np.random.default_rng(seed + 101 * (fold + 1))
    h_rows = sample_indices(len(comp["val_h"]), max_n, rng)
    m_rows = sample_indices(len(comp["val_m"]), max_n, rng)
    x_h_orth = gather_rows(comp["val_h"], h_rows, h_idx, bs=bs)
    x_m_orth = gather_rows(comp["val_m"], m_rows, m_idx, bs=bs)

    rows = []
    baseline = centroid_row(
        x_h_orth, x_m_orth, fold, "h2m", "liver_test_orthologues",
        "expression", "orthologues", "source_liver_test_vs_target_liver_test"
    )
    baseline.update(nearest_target_summary(x_h_orth, x_m_orth, k=k))
    rows.append(baseline)
    baseline = centroid_row(
        x_m_orth, x_h_orth, fold, "m2h", "liver_test_orthologues",
        "expression", "orthologues", "source_liver_test_vs_target_liver_test"
    )
    baseline.update(nearest_target_summary(x_m_orth, x_h_orth, k=k))
    rows.append(baseline)

    target_mouse_all = gather_rows(comp["val_m"], m_rows, cols=None, bs=bs)
    target_mouse_orth = target_mouse_all[:, m_idx]
    for mode, (enc, flow, dec) in translation_modes(comp, "h2m").items():
        pred_all = translate_rows(comp["val_h"], h_rows, enc, flow, dec, device, comp["Gm"], bs=bs)
        for feature_set, pred, target in [
            ("all_target_genes", pred_all, target_mouse_all),
            ("orthologues", pred_all[:, m_idx], target_mouse_orth),
        ]:
            row = centroid_row(
                pred, target, fold, "h2m", mode, "expression", feature_set,
                "translated_liver_source_vs_liver_target"
            )
            row.update(nearest_target_summary(pred, target, k=k))
            rows.append(row)
        del pred_all

    target_human_all = gather_rows(comp["val_h"], h_rows, cols=None, bs=bs)
    target_human_orth = target_human_all[:, h_idx]
    for mode, (enc, flow, dec) in translation_modes(comp, "m2h").items():
        pred_all = translate_rows(comp["val_m"], m_rows, enc, flow, dec, device, comp["Gh"], bs=bs)
        for feature_set, pred, target in [
            ("all_target_genes", pred_all, target_human_all),
            ("orthologues", pred_all[:, h_idx], target_human_orth),
        ]:
            row = centroid_row(
                pred, target, fold, "m2h", mode, "expression", feature_set,
                "translated_liver_source_vs_liver_target"
            )
            row.update(nearest_target_summary(pred, target, k=k))
            rows.append(row)
        del pred_all

    pd.DataFrame(rows).to_csv(
        EVAL_DIR / f"liver_centroid_expression_fold{fold}.csv", index=False
    )


def evaluate_fold(args, device, log):
    log(f"Fold {args.fold} | device={device}")
    comp = load_liver_components(args.fold, device)
    log(f"Liver samples: human={len(comp['val_h'])}, mouse={len(comp['val_m'])}")

    if not args.skip_cycle:
        cycle_liver(comp, args.fold, device, args.batch_size, log)
    if not args.skip_orthologue:
        orthologue_liver(comp, args.fold, device, args.batch_size, log)
    if not args.skip_latent_mmd:
        latent_mmd_liver(
            comp, args.fold, device, args.max_latent_mmd_samples,
            args.batch_size, args.seed, log,
            background_repeats=args.background_repeats,
            background_samples=args.background_samples
        )
    if not args.skip_expression_mmd:
        expression_mmd_liver(
            comp, args.fold, device, args.max_expression_mmd_samples,
            args.batch_size, args.seed, log,
            background_repeats=args.background_repeats,
            background_samples=args.background_samples
        )
    if not args.skip_liver_centroids:
        expression_centroid_liver(
            comp, args.fold, device, args.max_liver_centroid_samples,
            args.batch_size, args.seed, args.nearest_k, log
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--max_latent_mmd_samples", type=int, default=10000)
    parser.add_argument("--max_expression_mmd_samples", type=int, default=10000)
    parser.add_argument("--max_liver_centroid_samples", type=int, default=10000)
    parser.add_argument("--background_repeats", type=int, default=10,
                        help="Random non-liver background draws for liver-vs-other-tissue diagnostics.")
    parser.add_argument("--background_samples", type=int, default=1000,
                        help="Rows per random non-liver background draw.")
    parser.add_argument("--nearest_k", type=int, default=10)
    parser.add_argument("--skip_cycle", action="store_true")
    parser.add_argument("--skip_orthologue", action="store_true")
    parser.add_argument("--skip_latent_mmd", action="store_true")
    parser.add_argument("--skip_expression_mmd", action="store_true")
    parser.add_argument("--skip_liver_centroids", action="store_true")
    args = parser.parse_args()

    log = setup_logger(args.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_fold(args, device, log)


if __name__ == "__main__":
    main()
