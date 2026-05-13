#!/usr/bin/env python3
"""
Expression-space MMD evaluation for ARCHS4 FlowTransOP translations.

This script is cluster-oriented and intentionally not run by Codex locally.

Outputs one CSV per fold:
  ../archs4/evaluation/expression_mmd_fold{fold}.csv

Rows include:
  - validation_orthologues: MMD between source and target validation expression
    restricted to matched orthologues. This is the uncorrected expression-space
    baseline and is valid because both matrices have the same orthologue columns.
  - FlowTransOP / permuted_* translated outputs compared against target validation
    expression, either over target-side orthologues or all target genes.

For h2m, translated human validation expression is compared to mouse validation
expression. For m2h, translated mouse validation expression is compared to human
validation expression.
"""

import argparse
import logging
from logging import FileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import evaluate_translation as et


DATA_DIR = Path("../archs4")
EVAL_DIR = DATA_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def setup_logger(fold):
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger("expression_mmd")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    fh = FileHandler(f"logs/expression_mmd_fold{fold}.log", mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger.info


def sample_indices(n, max_n, rng):
    take = min(int(max_n), int(n))
    return np.sort(rng.choice(n, size=take, replace=False))


def gather_rows(X_lazy, rows, cols=None, bs=512):
    if cols is not None:
        cols = np.asarray(cols, dtype=np.int64)
        out = np.empty((len(rows), len(cols)), dtype=np.float32)
    else:
        out = np.empty((len(rows), X_lazy.shape[1]), dtype=np.float32)

    for start in range(0, len(rows), bs):
        stop = min(start + bs, len(rows))
        block = X_lazy[rows[start:stop], :].numpy()
        if cols is not None:
            block = block[:, cols]
        out[start:stop] = block
    return out


@torch.no_grad()
def translate_rows(X_lazy, rows, enc, flow, dec, device, out_dim, cols=None, bs=512):
    enc.eval()
    flow.eval()
    dec.eval()
    if cols is not None:
        cols = np.asarray(cols, dtype=np.int64)
        out = np.empty((len(rows), len(cols)), dtype=np.float32)
    else:
        out = np.empty((len(rows), out_dim), dtype=np.float32)

    for start in range(0, len(rows), bs):
        stop = min(start + bs, len(rows))
        X = X_lazy[rows[start:stop], :].to(device)
        z = enc(X)
        z_t = et.flow_step_n(flow, z)
        mu, _ = dec(z_t)
        mu_np = mu.cpu().numpy()
        if cols is not None:
            mu_np = mu_np[:, cols]
        out[start:stop] = mu_np
    return out


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


def mmd_value(X, Y, max_n, seed):
    return et.mmd_rbf(X, Y, max_n=max_n, seed=seed)


def evaluate_fold(fold, device, max_n, bs, seed, log):
    log(f"Fold {fold} | device={device} | max_n={max_n}")
    comp = et.load_all_components(fold, device)

    h_genes = et.load_genes("human")
    m_genes = et.load_genes("mouse")
    h_idx, m_idx = et.build_orthologue_map(h_genes, m_genes)
    log(f"Matched orthologues by case-insensitive symbol: {len(h_idx)}")

    rng = np.random.default_rng(seed + fold)
    h_rows = sample_indices(len(comp["val_h"]), max_n, rng)
    m_rows = sample_indices(len(comp["val_m"]), max_n, rng)

    rows = []

    # Untranslated expression-space baseline only for orthologues, where source
    # and target matrices share matched columns.
    x_h_orth = gather_rows(comp["val_h"], h_rows, h_idx, bs=bs)
    x_m_orth = gather_rows(comp["val_m"], m_rows, m_idx, bs=bs)
    baseline_mmd = mmd_value(x_h_orth, x_m_orth, max_n=max_n, seed=seed + fold)
    for direction in ["h2m", "m2h"]:
        for feature_set, comparison in [
            ("orthologues", "source_validation_vs_target_validation"),
            ("all_target_genes", "source_validation_vs_target_validation_orthologue_reference"),
        ]:
            rows.append({
                "fold": fold,
                "direction": direction,
                "feature_set": feature_set,
                "model_type": "validation_orthologues",
                "comparison": comparison,
                "n_source": len(h_rows) if direction == "h2m" else len(m_rows),
                "n_target": len(m_rows) if direction == "h2m" else len(h_rows),
                "n_features": len(h_idx),
                "mmd": baseline_mmd,
            })

    # h2m: translated human validation -> mouse expression.
    target_mouse_all = gather_rows(comp["val_m"], m_rows, cols=None, bs=bs)
    target_mouse_orth = target_mouse_all[:, m_idx]
    for model_type, (enc, flow, dec) in translation_modes(comp, "h2m").items():
        log(f"h2m {model_type}: translating sampled human validation rows")
        pred_all = translate_rows(comp["val_h"], h_rows, enc, flow, dec, device, comp["Gm"], cols=None, bs=bs)
        pred_orth = pred_all[:, m_idx]
        rows.append({
            "fold": fold,
            "direction": "h2m",
            "feature_set": "all_target_genes",
            "model_type": model_type,
            "comparison": "translated_source_validation_vs_target_validation",
            "n_source": len(h_rows),
            "n_target": len(m_rows),
            "n_features": comp["Gm"],
            "mmd": mmd_value(pred_all, target_mouse_all, max_n=max_n, seed=seed + fold),
        })
        rows.append({
            "fold": fold,
            "direction": "h2m",
            "feature_set": "orthologues",
            "model_type": model_type,
            "comparison": "translated_source_validation_vs_target_validation",
            "n_source": len(h_rows),
            "n_target": len(m_rows),
            "n_features": len(m_idx),
            "mmd": mmd_value(pred_orth, target_mouse_orth, max_n=max_n, seed=seed + fold),
        })
        del pred_all, pred_orth

    # m2h: translated mouse validation -> human expression.
    target_human_all = gather_rows(comp["val_h"], h_rows, cols=None, bs=bs)
    target_human_orth = target_human_all[:, h_idx]
    for model_type, (enc, flow, dec) in translation_modes(comp, "m2h").items():
        log(f"m2h {model_type}: translating sampled mouse validation rows")
        pred_all = translate_rows(comp["val_m"], m_rows, enc, flow, dec, device, comp["Gh"], cols=None, bs=bs)
        pred_orth = pred_all[:, h_idx]
        rows.append({
            "fold": fold,
            "direction": "m2h",
            "feature_set": "all_target_genes",
            "model_type": model_type,
            "comparison": "translated_source_validation_vs_target_validation",
            "n_source": len(m_rows),
            "n_target": len(h_rows),
            "n_features": comp["Gh"],
            "mmd": mmd_value(pred_all, target_human_all, max_n=max_n, seed=seed + fold),
        })
        rows.append({
            "fold": fold,
            "direction": "m2h",
            "feature_set": "orthologues",
            "model_type": model_type,
            "comparison": "translated_source_validation_vs_target_validation",
            "n_source": len(m_rows),
            "n_target": len(h_rows),
            "n_features": len(h_idx),
            "mmd": mmd_value(pred_orth, target_human_orth, max_n=max_n, seed=seed + fold),
        })
        del pred_all, pred_orth

    out = pd.DataFrame(rows)
    out_path = EVAL_DIR / f"expression_mmd_fold{fold}.csv"
    out.to_csv(out_path, index=False)
    log(f"Wrote {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--max_n", type=int, default=1000,
                        help="Rows sampled per source/target validation set for MMD.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()

    log = setup_logger(args.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_fold(args.fold, device, args.max_n, args.batch_size, args.seed, log)


if __name__ == "__main__":
    main()
