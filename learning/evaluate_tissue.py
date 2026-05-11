#!/usr/bin/env python3
"""
Tissue-level evaluation of FlowTransOP cross-species translation.

For each fold:
  1. Extract tissue labels for val samples from ARCHS4 metadata.
  2. Compute per-tissue centroids in latent space (target species real
     latents, source species translated latents).
  3. For each tissue T present in both species' val sets, measure:
       - Distance from translated-T centroid to its true target-T centroid.
       - Tissue rank: where does the correct target tissue land when sorting
         all target tissues by distance from translated_T? (rank 0 = nearest.)
       - Across-tissue centroid Pearson r (per latent dim, averaged).
  4. Same analysis for the permuted_both baseline (in its own latent space).

Output CSVs (under ../archs4/evaluation/):
  - tissue_assignments_fold{f}.csv  — tissue label per sample (for inspection)
  - tissue_metrics_fold{f}.csv      — per-tissue per-direction per-model metrics
  - tissue_summary_fold{f}.csv      — aggregated across tissues per direction/model

Tissue extraction is heuristic (case-insensitive keyword matching against a
fixed list of major tissues). It will leave many samples unlabeled — that's
fine, we only use samples that get a confident match. Inspect
`tissue_assignments_fold{f}.csv` to verify quality.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import logging
from logging import FileHandler
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import evaluate_translation as et

DATA_DIR    = Path("../archs4")
PREPROC_DIR = DATA_DIR / "preprocessed"
SPLITS_DIR  = DATA_DIR / "splits"
MODEL_DIR   = DATA_DIR / "models"
EVAL_DIR    = DATA_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

H5_PATHS = {
    'human': str(DATA_DIR / "human_gene_v2.latest.h5"),
    'mouse': str(DATA_DIR / "mouse_gene_v2.latest.h5"),
}


# --------------------------- tissue extraction ---------------------------

# Keywords are kept conservative; preferring fewer high-precision matches to
# many noisy ones. Order matters — more specific patterns first.
TISSUE_KEYWORDS = [
    ('liver',       ['liver', 'hepato', 'hepatic', 'hepg2', 'hep g2']),
    ('brain',       ['brain', 'cortex', 'cortical', 'hippocamp', 'cerebell',
                     'cerebrum', 'striatum', 'thalamus', 'amygdala', 'neuron', 'neural']),
    ('heart',       ['heart', 'cardiac', 'cardiomyo', 'ventricl', 'myocard']),
    ('lung',        ['lung', 'pulmonary', 'bronch', 'alveolar']),
    ('kidney',      ['kidney', 'renal', 'nephron', 'glomerul']),
    ('spleen',      ['spleen', 'splenic']),
    ('skeletal_muscle', ['skeletal muscle', 'gastrocnemius', 'quadriceps', 'soleus', 'tibialis']),
    ('skin',        ['skin', 'dermal', 'epidermal', 'keratinocyte']),
    ('adipose',     ['adipose', 'adipocyte']),
    ('blood',       [' blood', 'leukocyte', 'lymphocyte', 'monocyte', 'neutrophil',
                     'pbmc', 'peripheral blood', 't cell', 'b cell', 't-cell', 'b-cell']),
    ('bone_marrow', ['bone marrow', 'marrow']),
    ('thymus',      ['thymus', 'thymic']),
    ('intestine',   ['intestin', 'ileum', 'jejunum', 'duoden', 'colon']),
    ('stomach',     ['stomach', 'gastric']),
    ('pancreas',    ['pancreas', 'pancreatic', 'islet']),
    ('ovary',       ['ovary', 'ovarian']),
    ('testis',      ['testis', 'testicular']),
    ('breast',      ['breast', 'mammary']),
    ('prostate',    ['prostate', 'prostatic']),
    ('bladder',     ['bladder']),
    ('embryonic',   ['embryonic stem', 'ipsc', 'induced pluripotent']),
]


def assign_tissue(text):
    if text is None:
        return None
    s = text.lower()
    for tissue, keywords in TISSUE_KEYWORDS:
        for kw in keywords:
            if kw in s:
                return tissue
    return None


def get_val_tissues(species, fold, log):
    """Return (val_tissues, raw_strings) aligned with the val rows in the preprocessed memmap."""
    val_idx = np.load(PREPROC_DIR / f"{species}_fold{fold}_val_idx.npy")
    splits = json.load(open(SPLITS_DIR / f"{species}_split.json"))
    sorted_train_cols = np.sort(np.asarray(splits["train_indices"], dtype=np.int64))
    val_h5_cols = sorted_train_cols[val_idx]

    h5p = H5_PATHS[species]
    log(f"  reading metadata from {h5p}")
    with h5py.File(h5p, "r") as f:
        meta_samples = f["meta/samples"]
        chosen = None
        for field in ['tissue', 'source_name_ch1', 'characteristics_ch1', 'title']:
            if field in meta_samples:
                chosen = field
                break
        if chosen is None:
            raise RuntimeError(f"No tissue-relevant metadata field found for {species}")
        log(f"  using field: meta/samples/{chosen}")
        arr = meta_samples[chosen][:]

    if arr.dtype.kind in ('S', 'O'):
        raw = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in arr])
    else:
        raw = arr.astype(str)

    val_raw = raw[val_h5_cols]
    val_tissue = np.array([assign_tissue(t) for t in val_raw], dtype=object)
    n_assigned = sum(1 for t in val_tissue if t is not None)
    log(f"  {species} fold {fold}: {n_assigned}/{len(val_tissue)} samples assigned a tissue "
        f"({100*n_assigned/len(val_tissue):.1f}%)")
    return val_tissue, val_raw


# --------------------------- centroid analysis ---------------------------

def per_tissue_centroids(latents, tissues, valid_tissues, min_count):
    out, counts = {}, {}
    for t in valid_tissues:
        mask = (tissues == t)
        n = int(mask.sum())
        if n < min_count:
            continue
        out[t] = latents[mask].mean(axis=0)
        counts[t] = n
    return out, counts


def common_labels(y_ref, y_query, min_count):
    ref = {t for t in y_ref if t is not None and (y_ref == t).sum() >= min_count}
    query = {t for t in y_query if t is not None and (y_query == t).sum() >= min_count}
    return sorted(ref & query)


def subset_labeled(Z, y, labels):
    labels = set(labels)
    mask = np.array([(t in labels) for t in y], dtype=bool)
    return Z[mask], np.asarray(y[mask], dtype=str), np.where(mask)[0]


def sample_rows(Z, y, idx, max_n, rng):
    if max_n is None or max_n <= 0 or len(y) <= max_n:
        return Z, y, idx
    keep = rng.choice(len(y), size=max_n, replace=False)
    return Z[keep], y[keep], idx[keep]


def sample_labeled_indices(tissues, labels, max_n, rng):
    labels = set(labels)
    idx = np.where(np.array([(t in labels) for t in tissues], dtype=bool))[0]
    if max_n is not None and max_n > 0 and len(idx) > max_n:
        idx = np.sort(rng.choice(idx, size=max_n, replace=False))
    return idx


def safe_macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0))


def design_matrix(labels):
    labels = np.asarray(labels, dtype=str)
    levels = sorted(set(labels))
    mat = np.zeros((len(labels), len(levels)), dtype=np.float64)
    pos = {v: i for i, v in enumerate(levels)}
    for i, v in enumerate(labels):
        mat[i, pos[v]] = 1.0
    return mat, levels


def residual_sum_squares(Y, X):
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    return np.sum(resid ** 2, axis=0)


def get_translation_components(comp, direction, model_type):
    n, p = comp['normal'], comp['permuted']
    if direction == 'h2m':
        if model_type == 'FlowTransOP':
            return n['enc_h'], n['flow_h2m'], n['dec_m'], comp['val_h'], comp['Gm']
        if model_type == 'permuted_both':
            return p['enc_h'], p['flow_h2m'], p['dec_m'], comp['val_h'], comp['Gm']
    elif direction == 'm2h':
        if model_type == 'FlowTransOP':
            return n['enc_m'], n['flow_m2h'], n['dec_h'], comp['val_m'], comp['Gh']
        if model_type == 'permuted_both':
            return p['enc_m'], p['flow_m2h'], p['dec_h'], comp['val_m'], comp['Gh']
    raise ValueError(f"Unsupported direction/model_type: {direction}/{model_type}")


@torch.no_grad()
def translate_expression_rows(X_lazy, rows, enc, flow, dec, device, out_dim, cols=None, bs=256):
    enc.eval(); flow.eval(); dec.eval()
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


def gather_expression_rows(X_lazy, rows, cols=None, bs=1024):
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


def corr_1d(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def cosine_sim_matrix(A, B):
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return A @ B.T


def centroid_retrieval_expression(pred_expr, source_tissue, target_expr, target_tissue,
                                  direction, model_type, feature_set, fold, min_count):
    labels = common_labels(target_tissue, source_tissue, min_count)
    if not labels:
        return [], None
    pred_cent, n_source = per_tissue_centroids(pred_expr, source_tissue, labels, min_count)
    tgt_cent, n_target = per_tissue_centroids(target_expr, target_tissue, labels, min_count)
    labels = [t for t in labels if t in pred_cent and t in tgt_cent]
    if not labels:
        return [], None
    P = np.stack([pred_cent[t] for t in labels])
    T = np.stack([tgt_cent[t] for t in labels])
    cosine = cosine_sim_matrix(P, T)

    rows = []
    for i, tissue in enumerate(labels):
        pear = np.array([corr_1d(P[i], T[j]) for j in range(len(labels))], dtype=np.float64)
        spear = np.array([
            spearmanr(P[i], T[j]).correlation if np.std(P[i]) > 1e-12 and np.std(T[j]) > 1e-12 else np.nan
            for j in range(len(labels))
        ], dtype=np.float64)
        metrics = {'pearson': pear, 'spearman': spear, 'cosine': cosine[i]}
        for metric, sims in metrics.items():
            finite = np.where(np.isfinite(sims), sims, -np.inf)
            order = np.argsort(-finite)
            rank = int(np.where(order == i)[0][0])
            other = np.delete(sims, i)
            other = other[np.isfinite(other)]
            if len(other):
                margin = sims[i] - np.max(other)
                z = (sims[i] - np.mean(other)) / (np.std(other) + 1e-12)
            else:
                margin, z = np.nan, np.nan
            rows.append({
                'fold': fold, 'direction': direction, 'model_type': model_type,
                'feature_set': feature_set, 'metric': metric, 'tissue': tissue,
                'n_source': int(n_source[tissue]), 'n_target': int(n_target[tissue]),
                'similarity_to_correct': float(sims[i]),
                'rank': rank,
                'top1_correct': bool(rank == 0),
                'reciprocal_rank': float(1.0 / (rank + 1)),
                'similarity_margin': float(margin),
                'similarity_z': float(z),
            })
    summary = pd.DataFrame(rows).groupby('metric').agg(
        n_tissues=('tissue', 'nunique'),
        top1_accuracy=('top1_correct', 'mean'),
        mean_rank=('rank', 'mean'),
        mean_reciprocal_rank=('reciprocal_rank', 'mean'),
        mean_similarity_to_correct=('similarity_to_correct', 'mean'),
        mean_similarity_margin=('similarity_margin', 'mean'),
        mean_similarity_z=('similarity_z', 'mean'),
    ).reset_index()
    summary.insert(0, 'feature_set', feature_set)
    summary.insert(0, 'model_type', model_type)
    summary.insert(0, 'direction', direction)
    summary.insert(0, 'fold', fold)
    return rows, summary.to_dict('records')


def expression_knn_neighborhood(pred_expr, source_tissue, target_expr, target_tissue,
                                direction, model_type, feature_set, fold, k, chunk_size=256):
    labels = sorted(set(source_tissue) & set(target_tissue))
    if len(labels) < 2 or len(source_tissue) == 0 or len(target_tissue) == 0:
        return [], None
    kk = min(k, len(target_tissue))
    target_norm = target_expr / np.maximum(np.linalg.norm(target_expr, axis=1, keepdims=True), 1e-12)
    pred_norm = pred_expr / np.maximum(np.linalg.norm(pred_expr, axis=1, keepdims=True), 1e-12)
    rows = []
    target_tissue = np.asarray(target_tissue, dtype=str)
    source_tissue = np.asarray(source_tissue, dtype=str)
    for start in range(0, pred_norm.shape[0], chunk_size):
        stop = min(start + chunk_size, pred_norm.shape[0])
        sim = pred_norm[start:stop] @ target_norm.T
        top_idx = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
        top_sim = np.take_along_axis(sim, top_idx, axis=1)
        order = np.argsort(-top_sim, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_sim = np.take_along_axis(top_sim, order, axis=1)
        top_labels = target_tissue[top_idx]
        query_labels = source_tissue[start:stop]
        same = top_labels == query_labels[:, None]
        for i in range(stop - start):
            rows.append({
                'fold': fold, 'direction': direction, 'model_type': model_type,
                'feature_set': feature_set, 'sample_idx_in_sample': start + i,
                'tissue': query_labels[i],
                'top1_tissue': top_labels[i, 0],
                'top1_similarity': float(top_sim[i, 0]),
                'same_tissue_fraction': float(same[i].mean()),
                'top1_correct': bool(same[i, 0]),
                'topk_correct': bool(same[i].any()),
            })
    df = pd.DataFrame(rows)
    summary = {
        'fold': fold, 'direction': direction, 'model_type': model_type,
        'feature_set': feature_set, 'k': kk,
        'n_query': len(source_tissue), 'n_target': len(target_tissue),
        'mean_same_tissue_fraction': float(df['same_tissue_fraction'].mean()),
        'top1_accuracy': float(df['top1_correct'].mean()),
        'topk_accuracy': float(df['topk_correct'].mean()),
    }
    return rows, summary


def analyze_direction(z_target, t_target, z_translated, t_translated, log,
                      direction, model_type, fold, min_count):
    """Compute per-tissue centroid metrics. Returns (per_tissue_rows, summary_row)."""
    # np.unique can't sort object arrays containing None, so filter first.
    unique_src = {t for t in t_translated if t is not None}
    unique_tgt = {t for t in t_target if t is not None}
    src_tissues = [t for t in unique_src if (t_translated == t).sum() >= min_count]
    tgt_tissues = [t for t in unique_tgt if (t_target == t).sum() >= min_count]
    common = sorted(set(src_tissues) & set(tgt_tissues))
    if not common:
        log(f"  ({direction}, {model_type}): no tissue with >={min_count} samples in both species — skipping")
        return [], None

    cent_target, n_target = per_tissue_centroids(z_target, t_target, common, min_count)
    cent_trans,  n_trans  = per_tissue_centroids(z_translated, t_translated, common, min_count)
    common = [t for t in common if t in cent_target and t in cent_trans]
    if not common:
        return [], None

    target_mat = np.stack([cent_target[t] for t in common])    # (T, D)
    trans_mat  = np.stack([cent_trans[t]  for t in common])    # (T, D)

    target_norm = target_mat / np.maximum(np.linalg.norm(target_mat, axis=1, keepdims=True), 1e-12)
    trans_norm = trans_mat / np.maximum(np.linalg.norm(trans_mat, axis=1, keepdims=True), 1e-12)
    sim_mat = trans_norm @ target_norm.T

    rows, correct, cosine_correct = [], 0, 0
    for i, t in enumerate(common):
        dists = np.linalg.norm(target_mat - trans_mat[i], axis=1)
        order = np.argsort(dists)
        rank = int(np.where(order == i)[0][0])
        is_nn = (rank == 0)
        if is_nn:
            correct += 1
        sims = sim_mat[i]
        sim_order = np.argsort(-sims)
        cosine_rank = int(np.where(sim_order == i)[0][0])
        cosine_is_nn = (cosine_rank == 0)
        if cosine_is_nn:
            cosine_correct += 1
        cosine_z = (sims[i] - sims.mean()) / (sims.std() + 1e-12)
        rows.append({
            'fold': fold, 'direction': direction, 'model_type': model_type,
            'tissue': t,
            'n_source': int(n_trans[t]), 'n_target': int(n_target[t]),
            'distance_to_correct': float(dists[i]),
            'tissue_rank': rank,
            'nn_correct': bool(is_nn),
            'cosine_to_correct': float(sims[i]),
            'cosine_rank': cosine_rank,
            'cosine_nn_correct': bool(cosine_is_nn),
            'cosine_z': float(cosine_z),
        })

    # Per-dim Pearson across tissues, averaged
    pearsons = []
    for d in range(target_mat.shape[1]):
        if target_mat[:, d].std() < 1e-12 or trans_mat[:, d].std() < 1e-12:
            continue
        r, _ = pearsonr(target_mat[:, d], trans_mat[:, d])
        if not np.isnan(r):
            pearsons.append(r)
    centroid_corr = float(np.mean(pearsons)) if pearsons else float('nan')

    summary = {
        'fold': fold, 'direction': direction, 'model_type': model_type,
        'n_tissues': len(common),
        'mean_distance':           float(np.mean([r['distance_to_correct'] for r in rows])),
        'nn_accuracy':             correct / len(common),
        'mean_cosine_to_correct':  float(np.mean([r['cosine_to_correct'] for r in rows])),
        'mean_cosine_rank':        float(np.mean([r['cosine_rank'] for r in rows])),
        'cosine_nn_accuracy':      cosine_correct / len(common),
        'mean_cosine_z':           float(np.mean([r['cosine_z'] for r in rows])),
        'centroid_pearson_mean':   centroid_corr,
        'tissues':                 ';'.join(common),
    }
    log(f"  ({direction}, {model_type}): {len(common)} tissues, "
        f"NN-acc={summary['nn_accuracy']:.3f}, "
        f"cosNN-acc={summary['cosine_nn_accuracy']:.3f}, "
        f"<dist>={summary['mean_distance']:.4f}, "
        f"<cos>={summary['mean_cosine_to_correct']:.4f}, "
        f"centroid r={summary['centroid_pearson_mean']:.4f}")
    return rows, summary


def reference_mapping_knn(z_target, t_target, z_translated, t_translated, log,
                          direction, model_type, fold, min_count, k,
                          max_reference, max_query, rng):
    labels = common_labels(t_target, t_translated, min_count)
    if len(labels) < 2:
        log(f"  ({direction}, {model_type}, kNN): <2 common tissues — skipping")
        return None
    X_ref, y_ref, idx_ref = subset_labeled(z_target, t_target, labels)
    X_q, y_q, idx_q = subset_labeled(z_translated, t_translated, labels)
    X_ref, y_ref, idx_ref = sample_rows(X_ref, y_ref, idx_ref, max_reference, rng)
    X_q, y_q, idx_q = sample_rows(X_q, y_q, idx_q, max_query, rng)

    scaler = StandardScaler()
    X_ref_s = scaler.fit_transform(X_ref)
    X_q_s = scaler.transform(X_q)
    kk = min(k, len(y_ref))
    clf = KNeighborsClassifier(n_neighbors=kk, weights='distance')
    clf.fit(X_ref_s, y_ref)
    pred = clf.predict(X_q_s)
    row = {
        'fold': fold, 'direction': direction, 'model_type': model_type,
        'method': f'knn_reference_k{kk}',
        'n_tissues': len(labels), 'n_reference': len(y_ref), 'n_query': len(y_q),
        'macro_f1': safe_macro_f1(y_q, pred),
        'balanced_accuracy': float(balanced_accuracy_score(y_q, pred)),
        'accuracy': float(accuracy_score(y_q, pred)),
        'tissues': ';'.join(labels),
    }
    log(f"  ({direction}, {model_type}, kNN ref): macroF1={row['macro_f1']:.3f}, "
        f"balAcc={row['balanced_accuracy']:.3f}, n={len(y_q)}")
    return row


def linear_probe(z_target, t_target, z_translated, t_translated, log,
                 direction, model_type, fold, min_count, max_reference,
                 max_query, seed, rng):
    labels = common_labels(t_target, t_translated, min_count)
    if len(labels) < 2:
        log(f"  ({direction}, {model_type}, linear): <2 common tissues — skipping")
        return None
    X_ref, y_ref, idx_ref = subset_labeled(z_target, t_target, labels)
    X_q, y_q, idx_q = subset_labeled(z_translated, t_translated, labels)
    X_ref, y_ref, idx_ref = sample_rows(X_ref, y_ref, idx_ref, max_reference, rng)
    X_q, y_q, idx_q = sample_rows(X_q, y_q, idx_q, max_query, rng)

    scaler = StandardScaler()
    X_ref_s = scaler.fit_transform(X_ref)
    X_q_s = scaler.transform(X_q)
    clf = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=1e-4,
        class_weight='balanced',
        max_iter=1000,
        tol=1e-3,
        random_state=seed,
    )
    clf.fit(X_ref_s, y_ref)
    pred = clf.predict(X_q_s)
    row = {
        'fold': fold, 'direction': direction, 'model_type': model_type,
        'method': 'linear_probe_sgd_logistic',
        'n_tissues': len(labels), 'n_reference': len(y_ref), 'n_query': len(y_q),
        'macro_f1': safe_macro_f1(y_q, pred),
        'balanced_accuracy': float(balanced_accuracy_score(y_q, pred)),
        'accuracy': float(accuracy_score(y_q, pred)),
        'tissues': ';'.join(labels),
    }
    log(f"  ({direction}, {model_type}, linear): macroF1={row['macro_f1']:.3f}, "
        f"balAcc={row['balanced_accuracy']:.3f}, n={len(y_q)}")
    return row


def knn_tissue_purity(z_target, t_target, z_translated, t_translated, log,
                      direction, model_type, fold, min_count, k,
                      max_reference, max_query, rng):
    labels = common_labels(t_target, t_translated, min_count)
    if len(labels) < 2:
        log(f"  ({direction}, {model_type}, purity): <2 common tissues — skipping")
        return None
    X_ref, y_ref, idx_ref = subset_labeled(z_target, t_target, labels)
    X_q, y_q, idx_q = subset_labeled(z_translated, t_translated, labels)
    X_ref, y_ref, idx_ref = sample_rows(X_ref, y_ref, idx_ref, max_reference, rng)
    X_q, y_q, idx_q = sample_rows(X_q, y_q, idx_q, max_query, rng)

    scaler = StandardScaler()
    X_ref_s = scaler.fit_transform(X_ref)
    X_q_s = scaler.transform(X_q)
    kk = min(k, len(y_ref))
    nn = NearestNeighbors(n_neighbors=kk)
    nn.fit(X_ref_s)
    _, nbr_idx = nn.kneighbors(X_q_s)
    nbr_labels = y_ref[nbr_idx]
    same = nbr_labels == y_q[:, None]
    top1 = same[:, 0]
    topk = same.any(axis=1)
    purity = same.mean(axis=1)

    row = {
        'fold': fold, 'direction': direction, 'model_type': model_type,
        'method': f'knn_purity_k{kk}',
        'n_tissues': len(labels), 'n_reference': len(y_ref), 'n_query': len(y_q),
        'mean_same_tissue_fraction': float(purity.mean()),
        'top1_accuracy': float(top1.mean()),
        'topk_accuracy': float(topk.mean()),
        'tissues': ';'.join(labels),
    }
    log(f"  ({direction}, {model_type}, purity): sameFrac={row['mean_same_tissue_fraction']:.3f}, "
        f"top1={row['top1_accuracy']:.3f}, top{kk}={row['topk_accuracy']:.3f}")
    return row


def variance_partitioning(z_target, t_target, z_translated, t_translated, log,
                          direction, model_type, fold, min_count, n_pcs,
                          max_samples, rng):
    labels = common_labels(t_target, t_translated, min_count)
    if len(labels) < 2:
        log(f"  ({direction}, {model_type}, variance): <2 common tissues — skipping")
        return [], None
    X_ref, y_ref, idx_ref = subset_labeled(z_target, t_target, labels)
    X_q, y_q, idx_q = subset_labeled(z_translated, t_translated, labels)
    X = np.vstack([X_ref, X_q])
    tissue = np.concatenate([y_ref, y_q])
    domain = np.array(['target_real'] * len(y_ref) + ['translated_source'] * len(y_q))
    idx_all = np.arange(len(tissue))
    X, tissue, idx_all = sample_rows(X, tissue, idx_all, max_samples, rng)
    domain = domain[idx_all]

    Xs = StandardScaler().fit_transform(X)
    n_comp = min(n_pcs, Xs.shape[1], Xs.shape[0] - 1)
    if n_comp < 1:
        return [], None
    pca = PCA(n_components=n_comp, random_state=0)
    scores = pca.fit_transform(Xs)

    intercept = np.ones((len(tissue), 1))
    tissue_X, tissue_levels = design_matrix(tissue)
    domain_X, domain_levels = design_matrix(domain)
    full_X = np.hstack([intercept, tissue_X, domain_X])
    no_tissue_X = np.hstack([intercept, domain_X])
    no_domain_X = np.hstack([intercept, tissue_X])
    rss_full = residual_sum_squares(scores, full_X)
    rss_no_tissue = residual_sum_squares(scores, no_tissue_X)
    rss_no_domain = residual_sum_squares(scores, no_domain_X)

    tissue_r2 = np.maximum((rss_no_tissue - rss_full) / np.maximum(rss_no_tissue, 1e-12), 0)
    domain_r2 = np.maximum((rss_no_domain - rss_full) / np.maximum(rss_no_domain, 1e-12), 0)
    weights = pca.explained_variance_ratio_ / np.maximum(pca.explained_variance_ratio_.sum(), 1e-12)

    rows = []
    for pc in range(n_comp):
        rows.append({
            'fold': fold, 'direction': direction, 'model_type': model_type,
            'pc': pc + 1,
            'explained_variance_ratio': float(pca.explained_variance_ratio_[pc]),
            'partial_r2_tissue': float(tissue_r2[pc]),
            'partial_r2_domain': float(domain_r2[pc]),
        })
    summary = {
        'fold': fold, 'direction': direction, 'model_type': model_type,
        'n_samples': len(tissue), 'n_tissues': len(labels), 'n_pcs': n_comp,
        'weighted_partial_r2_tissue': float(np.sum(weights * tissue_r2)),
        'weighted_partial_r2_domain': float(np.sum(weights * domain_r2)),
        'mean_partial_r2_tissue': float(np.mean(tissue_r2)),
        'mean_partial_r2_domain': float(np.mean(domain_r2)),
        'tissues': ';'.join(labels),
    }
    log(f"  ({direction}, {model_type}, variance): tissueR2={summary['weighted_partial_r2_tissue']:.3f}, "
        f"domainR2={summary['weighted_partial_r2_domain']:.3f}, n={len(tissue)}")
    return rows, summary


# --------------------------- main ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--min_count", type=int, default=20,
                        help="Minimum samples per tissue per species to include.")
    parser.add_argument("--knn_k", type=int, default=10,
                        help="k for reference mapping and neighborhood purity.")
    parser.add_argument("--max_reference_samples", type=int, default=50000,
                        help="Maximum real target samples used for kNN/probe references; <=0 uses all.")
    parser.add_argument("--max_query_samples", type=int, default=20000,
                        help="Maximum translated source samples evaluated by kNN/probes; <=0 uses all.")
    parser.add_argument("--variance_pcs", type=int, default=20,
                        help="Number of PCs used for variance partitioning.")
    parser.add_argument("--max_variance_samples", type=int, default=50000,
                        help="Maximum combined samples used for variance partitioning; <=0 uses all.")
    parser.add_argument("--max_expression_samples", type=int, default=10000,
                        help="Maximum source/target samples per direction for expression-space tissue metrics.")
    parser.add_argument("--expression_feature_sets", nargs='+',
                        default=['orthologues', 'all_target_genes'],
                        choices=['orthologues', 'all_target_genes'],
                        help="Feature sets for expression-space tissue centroid retrieval.")
    parser.add_argument("--expression_knn_feature_sets", nargs='+',
                        default=['orthologues', 'all_target_genes'],
                        choices=['orthologues', 'all_target_genes'],
                        help="Feature sets for sample-level expression-space kNN enrichment.")
    parser.add_argument("--expression_batch_size", type=int, default=256,
                        help="Batch size for decoding translated expression.")
    parser.add_argument("--expression_knn_chunk_size", type=int, default=128,
                        help="Query chunk size for expression-space cosine kNN.")
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    log_file = f'logs/evaluate_tissue_fold{args.fold}.log'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = FileHandler(log_file, mode='a'); fh.setFormatter(logging.Formatter('%(message)s'))
    sh = logging.StreamHandler();           sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh); logger.addHandler(sh)
    log = logger.info

    log(f"=== Tissue evaluation: fold {args.fold} (min_count={args.min_count}) ===")

    log("\nExtracting human val tissues…")
    t_human, raw_h = get_val_tissues("human", args.fold, log)
    log("Extracting mouse val tissues…")
    t_mouse, raw_m = get_val_tissues("mouse", args.fold, log)

    # Save tissue assignments (so you can inspect what got matched and what didn't)
    pd.DataFrame({
        'species':           ['human'] * len(t_human) + ['mouse'] * len(t_mouse),
        'sample_idx_in_val': np.concatenate([np.arange(len(t_human)),
                                              np.arange(len(t_mouse))]),
        'raw_metadata':      np.concatenate([raw_h, raw_m]),
        'tissue':            np.concatenate([t_human, t_mouse]),
    }).to_csv(EVAL_DIR / f"tissue_assignments_fold{args.fold}.csv", index=False)

    log("\nLoading latents…")
    z_h_val        = np.load(MODEL_DIR / f"fold_{args.fold}_z_human_val.npy")
    z_m_val        = np.load(MODEL_DIR / f"fold_{args.fold}_z_mouse_val.npy")
    z_h2m_val      = np.load(MODEL_DIR / f"fold_{args.fold}_z_h2m_val.npy")
    z_m2h_val      = np.load(MODEL_DIR / f"fold_{args.fold}_z_m2h_val.npy")
    z_h_val_perm   = np.load(MODEL_DIR / f"fold_{args.fold}_z_human_val_perm.npy")
    z_m_val_perm   = np.load(MODEL_DIR / f"fold_{args.fold}_z_mouse_val_perm.npy")
    z_h2m_val_perm = np.load(MODEL_DIR / f"fold_{args.fold}_z_h2m_val_perm.npy")
    z_m2h_val_perm = np.load(MODEL_DIR / f"fold_{args.fold}_z_m2h_val_perm.npy")

    log("\nLoading model components for expression-space tissue evaluation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    comp = et.load_all_components(args.fold, device)
    h_genes = et.load_genes("human")
    m_genes = et.load_genes("mouse")
    h_idx, m_idx = et.build_orthologue_map(h_genes, m_genes)
    log(f"  device={device}; matched orthologues={len(h_idx)}")

    all_rows, summaries = [], []
    reference_rows, linear_rows, purity_rows = [], [], []
    variance_rows, variance_summaries = [], []
    expr_centroid_rows, expr_centroid_summaries = [], []
    expr_knn_rows, expr_knn_summaries = [], []

    jobs = [
        {
            'direction': 'h2m', 'model_type': 'FlowTransOP',
            'z_target': z_m_val, 't_target': t_mouse,
            'z_translated': z_h2m_val, 't_translated': t_human,
        },
        {
            'direction': 'm2h', 'model_type': 'FlowTransOP',
            'z_target': z_h_val, 't_target': t_human,
            'z_translated': z_m2h_val, 't_translated': t_mouse,
        },
        {
            'direction': 'h2m', 'model_type': 'permuted_both',
            'z_target': z_m_val_perm, 't_target': t_mouse,
            'z_translated': z_h2m_val_perm, 't_translated': t_human,
        },
        {
            'direction': 'm2h', 'model_type': 'permuted_both',
            'z_target': z_h_val_perm, 't_target': t_human,
            'z_translated': z_m2h_val_perm, 't_translated': t_mouse,
        },
    ]

    rng = np.random.default_rng(args.seed + args.fold)
    for job in jobs:
        log(f"\n--- {job['model_type']} {job['direction']} ---")
        rows, s = analyze_direction(
            z_target=job['z_target'], t_target=job['t_target'],
            z_translated=job['z_translated'], t_translated=job['t_translated'],
            log=log, direction=job['direction'], model_type=job['model_type'],
            fold=args.fold, min_count=args.min_count)
        all_rows += rows
        if s:
            summaries.append(s)

        row = reference_mapping_knn(
            z_target=job['z_target'], t_target=job['t_target'],
            z_translated=job['z_translated'], t_translated=job['t_translated'],
            log=log, direction=job['direction'], model_type=job['model_type'],
            fold=args.fold, min_count=args.min_count, k=args.knn_k,
            max_reference=args.max_reference_samples, max_query=args.max_query_samples,
            rng=rng)
        if row:
            reference_rows.append(row)

        row = linear_probe(
            z_target=job['z_target'], t_target=job['t_target'],
            z_translated=job['z_translated'], t_translated=job['t_translated'],
            log=log, direction=job['direction'], model_type=job['model_type'],
            fold=args.fold, min_count=args.min_count,
            max_reference=args.max_reference_samples, max_query=args.max_query_samples,
            seed=args.seed + args.fold, rng=rng)
        if row:
            linear_rows.append(row)

        row = knn_tissue_purity(
            z_target=job['z_target'], t_target=job['t_target'],
            z_translated=job['z_translated'], t_translated=job['t_translated'],
            log=log, direction=job['direction'], model_type=job['model_type'],
            fold=args.fold, min_count=args.min_count, k=args.knn_k,
            max_reference=args.max_reference_samples, max_query=args.max_query_samples,
            rng=rng)
        if row:
            purity_rows.append(row)

        rows, s = variance_partitioning(
            z_target=job['z_target'], t_target=job['t_target'],
            z_translated=job['z_translated'], t_translated=job['t_translated'],
            log=log, direction=job['direction'], model_type=job['model_type'],
            fold=args.fold, min_count=args.min_count, n_pcs=args.variance_pcs,
            max_samples=args.max_variance_samples, rng=rng)
        variance_rows += rows
        if s:
            variance_summaries.append(s)

    expression_jobs = [
        {'direction': 'h2m', 'model_type': 'FlowTransOP',
         'source_tissue': t_human, 'target_tissue': t_mouse,
         'target_lazy': comp['val_m'], 'target_dim': comp['Gm'], 'orth_cols': m_idx},
        {'direction': 'm2h', 'model_type': 'FlowTransOP',
         'source_tissue': t_mouse, 'target_tissue': t_human,
         'target_lazy': comp['val_h'], 'target_dim': comp['Gh'], 'orth_cols': h_idx},
        {'direction': 'h2m', 'model_type': 'permuted_both',
         'source_tissue': t_human, 'target_tissue': t_mouse,
         'target_lazy': comp['val_m'], 'target_dim': comp['Gm'], 'orth_cols': m_idx},
        {'direction': 'm2h', 'model_type': 'permuted_both',
         'source_tissue': t_mouse, 'target_tissue': t_human,
         'target_lazy': comp['val_h'], 'target_dim': comp['Gh'], 'orth_cols': h_idx},
    ]

    for job in expression_jobs:
        log(f"\n--- expression-space tissue {job['model_type']} {job['direction']} ---")
        labels = common_labels(job['target_tissue'], job['source_tissue'], args.min_count)
        if len(labels) < 2:
            log("  <2 common tissues - skipping")
            continue
        source_rows = sample_labeled_indices(job['source_tissue'], labels, args.max_expression_samples, rng)
        target_rows = sample_labeled_indices(job['target_tissue'], labels, args.max_expression_samples, rng)
        source_y = np.asarray(job['source_tissue'][source_rows], dtype=str)
        target_y = np.asarray(job['target_tissue'][target_rows], dtype=str)
        enc, flow, dec, source_lazy, out_dim = get_translation_components(comp, job['direction'], job['model_type'])
        needed_sets = sorted(set(args.expression_feature_sets) | set(args.expression_knn_feature_sets))
        for feature_set in needed_sets:
            cols = None if feature_set == 'all_target_genes' else job['orth_cols']
            log(f"  feature_set={feature_set}: translating {len(source_rows)} source and gathering {len(target_rows)} target samples")
            pred_expr = translate_expression_rows(
                source_lazy, source_rows, enc, flow, dec, device, out_dim,
                cols=cols, bs=args.expression_batch_size)
            target_expr = gather_expression_rows(job['target_lazy'], target_rows, cols=cols)

            if feature_set in args.expression_feature_sets:
                rows, summary = centroid_retrieval_expression(
                    pred_expr, source_y, target_expr, target_y,
                    direction=job['direction'], model_type=job['model_type'],
                    feature_set=feature_set, fold=args.fold, min_count=args.min_count)
                expr_centroid_rows += rows
                expr_centroid_summaries += summary or []
                if summary:
                    for s in summary:
                        log(f"    centroid {s['metric']}: top1={s['top1_accuracy']:.3f}, "
                            f"MRR={s['mean_reciprocal_rank']:.3f}, z={s['mean_similarity_z']:.3f}")

            if feature_set in args.expression_knn_feature_sets:
                rows, summary = expression_knn_neighborhood(
                    pred_expr, source_y, target_expr, target_y,
                    direction=job['direction'], model_type=job['model_type'],
                    feature_set=feature_set, fold=args.fold, k=args.knn_k,
                    chunk_size=args.expression_knn_chunk_size)
                expr_knn_rows += rows
                if summary:
                    expr_knn_summaries.append(summary)
                    log(f"    sample kNN: sameFrac={summary['mean_same_tissue_fraction']:.3f}, "
                        f"top1={summary['top1_accuracy']:.3f}, top{summary['k']}={summary['topk_accuracy']:.3f}")

            del pred_expr, target_expr

    pd.DataFrame(all_rows).to_csv(
        EVAL_DIR / f"tissue_metrics_fold{args.fold}.csv", index=False)
    pd.DataFrame([s for s in summaries if s]).to_csv(
        EVAL_DIR / f"tissue_summary_fold{args.fold}.csv", index=False)
    pd.DataFrame(reference_rows).to_csv(
        EVAL_DIR / f"tissue_reference_mapping_fold{args.fold}.csv", index=False)
    pd.DataFrame(linear_rows).to_csv(
        EVAL_DIR / f"tissue_linear_probe_fold{args.fold}.csv", index=False)
    pd.DataFrame(purity_rows).to_csv(
        EVAL_DIR / f"tissue_knn_purity_fold{args.fold}.csv", index=False)
    pd.DataFrame(variance_rows).to_csv(
        EVAL_DIR / f"tissue_variance_partition_fold{args.fold}.csv", index=False)
    pd.DataFrame(variance_summaries).to_csv(
        EVAL_DIR / f"tissue_variance_summary_fold{args.fold}.csv", index=False)
    pd.DataFrame(expr_centroid_rows).to_csv(
        EVAL_DIR / f"tissue_expression_centroid_fold{args.fold}.csv", index=False)
    pd.DataFrame(expr_centroid_summaries).to_csv(
        EVAL_DIR / f"tissue_expression_centroid_summary_fold{args.fold}.csv", index=False)
    pd.DataFrame(expr_knn_rows).to_csv(
        EVAL_DIR / f"tissue_expression_knn_fold{args.fold}.csv", index=False)
    pd.DataFrame(expr_knn_summaries).to_csv(
        EVAL_DIR / f"tissue_expression_knn_summary_fold{args.fold}.csv", index=False)
    log(f"\n✓ Fold {args.fold} tissue evaluation complete.")


if __name__ == "__main__":
    main()
