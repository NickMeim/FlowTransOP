#!/usr/bin/env python3
"""
Score ARCHS4 liver subsets for NAS/MAS activity and fibrosis.

This script uses the preprocessed liver test matrices written by the ARCHS4
preprocessing scripts and the trained FlowTransOP fold checkpoints. It:

1. Selects the human Govaere et al. liver subset (GSE135251) from metadata,
   parses NAS and fibrosis labels, centers the expression matrix, and fits a
   two-output PLSR scorer with 8 latent variables.
2. Selects mouse Nlrp3A350V liver samples treated with Control chow+placebo
   or GS-444217, translates them to human expression space with FlowTransOP,
   centers them with the Govaere human reference mean, and predicts NAS/MAS
   and fibrosis scores with the same PLSR model.
3. Scores the same human and translated-mouse matrices with signed,
   rank-based NAS/MAS and fibrosis gene signatures inspired by VIPER/aREA.

Outputs are CSVs under ../archs4/evaluation/liver_mas_fibrosis by default.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

import evaluate_translation as et


DATA_DIR = Path("../archs4")
SPLITS_DIR = DATA_DIR / "splits"
PREPROC_DIR = DATA_DIR / "preprocessed"
MODEL_DIR = DATA_DIR / "models"
OUT_DIR = DATA_DIR / "evaluation" / "liver_mas_fibrosis"

HUMAN_GSE = "GSE135251"
MOUSE_GSE = "GSE196908"
PLS_COMPONENTS = 8

SIGNATURES = {
    "MAS": {
        "targets": [
            "ARPC5", "YWHAH", "ARF4", "TNFRSF12A", "ADHFE1",
            "USP33", "CD52", "ACVR2B", "ING5", "ASB3", "IFI30",
            "ERVW-1", "YWHAZ", "ERBB3", "KPNA2", "COQ10B", "MAGI1",
            "MAPRE1", "ABCA6",
        ],
        "mor": [1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1],
    },
    "Fibrosis stage": {
        "targets": [
            "TIMP1", "MYLI2B", "LUM", "ZNF395", "AKAP9",
            "ACTR2", "LGALS3", "MAPRE1", "FRK", "ANKRD28",
            "IGFBP7", "YWHAZ", "USP33", "CD59", "TAX1BP3",
            "FAM221A", "ADHFE1", "TNFRSF12A",
        ],
        "mor": [1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1],
    },
}


def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return path


def read_metadata(path):
    df = pd.read_csv(require_file(path), index_col=0)
    df.index = df.index.astype(str)
    return df.fillna("")


def series_mask(df, accession):
    """Use explicit series/GSE columns when they are present."""
    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        col_l = col.lower()
        if "series" in col_l or "gse" in col_l or "geo" in col_l:
            mask |= df[col].astype(str).str.contains(accession, case=False, na=False)
    return mask


def parse_characteristic_value(text, labels):
    for label in labels:
        pattern = rf"(?:^|,)\s*{re.escape(label)}\s*:\s*([^,]+)"
        match = re.search(pattern, str(text), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def parse_numeric_score(value):
    if value is None:
        return np.nan
    value = str(value).strip()
    if not value or value.lower() in {"na", "nan", "none", "not available"}:
        return np.nan
    if "normal liver histology" in value.lower():
        return 0.0
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else np.nan


def add_human_scores(meta):
    chars = meta["characteristics_ch1"].astype(str)
    out = meta.copy()
    out["nas_score"] = [
        parse_numeric_score(
            parse_characteristic_value(x, ["nas score", "nafld activity score"])
        )
        for x in chars
    ]
    out["fibrosis_stage"] = [
        parse_numeric_score(parse_characteristic_value(x, ["fibrosis stage"]))
        for x in chars
    ]
    return out


def select_human_govaere(meta, human_test_ids, accession=HUMAN_GSE):
    ids_in_matrix = pd.Index(human_test_ids.astype(str))
    meta = meta.loc[meta.index.intersection(ids_in_matrix)].copy()

    mask = series_mask(meta, accession)
    if not mask.any():
        chars = meta["characteristics_ch1"].astype(str)
        mask = (
            chars.str.contains(r"\bnas score\s*:", case=False, regex=True, na=False)
            & chars.str.contains(r"\bfibrosis stage\s*:", case=False, regex=True, na=False)
            & chars.str.contains(r"\bgroup in paper\s*:", case=False, regex=True, na=False)
        )

    selected = add_human_scores(meta.loc[mask].copy())
    selected = selected.dropna(subset=["nas_score", "fibrosis_stage"])
    if selected.empty:
        raise ValueError(
            "No scored Govaere human samples found. Expected explicit GSE135251 "
            "metadata or characteristics_ch1 with nas score, fibrosis stage, "
            "and group in paper."
        )
    return selected


def select_mouse_nlrp3(meta, mouse_test_ids, accession=MOUSE_GSE):
    ids_in_matrix = pd.Index(mouse_test_ids.astype(str))
    meta = meta.loc[meta.index.intersection(ids_in_matrix)].copy()
    chars = meta["characteristics_ch1"].astype(str)

    gse = series_mask(meta, accession)
    if not gse.any():
        gse = pd.Series(True, index=meta.index)

    genotype = chars.str.contains("Nlrp3A350V", case=False, regex=False, na=False)
    control = chars.str.contains("Control chow+placebo", case=False, regex=False, na=False)
    selonsertib = chars.str.contains("GS-444217", case=False, regex=False, na=False)
    selected = meta.loc[gse & genotype & (control | selonsertib)].copy()

    selected["mouse_genotype"] = "Nlrp3A350V"
    selected["mouse_treatment"] = np.where(
        selected["characteristics_ch1"].astype(str).str.contains(
            "GS-444217", case=False, regex=False, na=False
        ),
        "GS-444217",
        "Control chow+placebo",
    )
    if selected.empty:
        raise ValueError(
            "No mouse Nlrp3A350V samples found for Control chow+placebo or "
            "GS-444217 in the liver metadata."
        )
    return selected


def row_lookup(sample_ids):
    return {str(sample_id): i for i, sample_id in enumerate(sample_ids.astype(str))}


def rows_for_ids(requested_ids, sample_ids, label):
    lookup = row_lookup(sample_ids)
    missing = [sid for sid in requested_ids if str(sid) not in lookup]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"{len(missing)} {label} sample IDs are missing from matrix IDs: {preview}")
    return np.asarray([lookup[str(sid)] for sid in requested_ids], dtype=np.int64)


def load_rows(matrix, rows):
    return np.asarray(matrix[rows, :], dtype=np.float32)


def load_flowtransop_m2h(fold, human_dim, mouse_dim, model_dir, device):
    normal_path = require_file(model_dir / f"fold_{fold}_normal.pt")
    m2h_path = require_file(model_dir / f"fold_{fold}_normal_m2h.pt")

    ckpt = torch.load(normal_path, map_location=device, weights_only=False)
    ckpt_m2h = torch.load(m2h_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    _, enc_m, dec_h, _, _ = et.build_models(args, human_dim, mouse_dim, device)
    enc_m.load_state_dict(ckpt["encoder_mouse"])
    dec_h.load_state_dict(ckpt["decoder_human"])

    flow_m2h = et.Flow(args["latent_dim"], args["latent_dim"] // 2, dtype=torch.float).to(device)
    flow_m2h.load_state_dict(ckpt_m2h["flow_m2h"])

    enc_m.eval()
    flow_m2h.eval()
    dec_h.eval()
    return enc_m, flow_m2h, dec_h


@torch.no_grad()
def translate_mouse_to_human(mouse_matrix, mouse_rows, enc_m, flow_m2h, dec_h, device, batch_size):
    out_dim = int(dec_h.out_mu.out_features)
    translated = np.empty((len(mouse_rows), out_dim), dtype=np.float32)
    for start in range(0, len(mouse_rows), batch_size):
        stop = min(start + batch_size, len(mouse_rows))
        block = np.asarray(mouse_matrix[mouse_rows[start:stop], :], dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(block)).to(device)
        z = enc_m(x)
        z_h = et.flow_step_n(flow_m2h, z)
        mu, _ = dec_h(z_h)
        translated[start:stop] = mu.cpu().numpy()
    return translated


def fit_plsr_scorer(X_centered, y, n_components):
    max_components = min(n_components, X_centered.shape[0] - 1, X_centered.shape[1])
    if max_components < 1:
        raise ValueError("Not enough human samples/features to fit a PLSR model.")
    if max_components != n_components:
        print(f"Using {max_components} PLS components because the data are smaller than requested.")

    model = PLSRegression(n_components=max_components, scale=False)
    model.fit(X_centered, y)
    return model, max_components


def signature_regulon_frame():
    rows = []
    for source, spec in SIGNATURES.items():
        for target, mor in zip(spec["targets"], spec["mor"]):
            rows.append({"source": source, "target": target, "mor": int(mor)})
    return pd.DataFrame(rows)


def _gene_symbol_key(symbol):
    return str(symbol).strip().upper()


def map_signature_targets(human_genes):
    gene_to_idx = {}
    for idx, gene in enumerate(human_genes.astype(str)):
        gene_to_idx.setdefault(_gene_symbol_key(gene), idx)

    mapped = {}
    for source, spec in SIGNATURES.items():
        targets = spec["targets"]
        mor = np.asarray(spec["mor"], dtype=np.float32)
        idx, used_mor, matched, missing = [], [], [], []
        for target, target_mor in zip(targets, mor):
            gene_idx = gene_to_idx.get(_gene_symbol_key(target))
            if gene_idx is None:
                missing.append(target)
            else:
                idx.append(gene_idx)
                used_mor.append(float(target_mor))
                matched.append(target)
        if not idx:
            raise ValueError(f"No genes from the {source} signature are present in human_genes.npy.")
        mapped[source] = {
            "idx": np.asarray(idx, dtype=np.int64),
            "mor": np.asarray(used_mor, dtype=np.float32),
            "matched": matched,
            "missing": missing,
        }
    return mapped


def signed_rank_signature_activity(X_centered, human_genes):
    """VIPER/aREA-inspired signed rank enrichment per sample.

    Each sample is converted to standardized within-sample gene ranks. Signature
    activity is the signed mean rank of regulon targets, scaled by sqrt(n) so
    larger target sets are comparable to smaller target sets.
    """
    if X_centered.shape[1] != len(human_genes):
        raise ValueError(
            f"Expression has {X_centered.shape[1]} features, but human_genes.npy "
            f"has {len(human_genes)} genes."
        )

    mapped = map_signature_targets(human_genes)
    order = np.argsort(X_centered, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    sample_ids = np.arange(X_centered.shape[0])[:, None]
    ranks[sample_ids, order] = np.arange(X_centered.shape[1], dtype=np.float32)

    rank_mean = (X_centered.shape[1] - 1) / 2.0
    rank_sd = np.sqrt((X_centered.shape[1] ** 2 - 1) / 12.0)
    rank_z = (ranks - rank_mean) / rank_sd

    activity = {}
    meta_rows = []
    for source, info in mapped.items():
        signed = rank_z[:, info["idx"]] @ info["mor"]
        activity[source] = signed / np.sqrt(len(info["idx"]))
        meta_rows.append(
            {
                "signature": source,
                "n_targets_total": len(SIGNATURES[source]["targets"]),
                "n_targets_used": int(len(info["idx"])),
                "missing_targets": ";".join(info["missing"]),
            }
        )
    return pd.DataFrame(activity), pd.DataFrame(meta_rows)


def calibrate_signature_predictions(human_activity, mouse_activity, y_human):
    specs = [
        ("MAS", "nas_score", 0),
        ("Fibrosis stage", "fibrosis_stage", 1),
    ]
    human_pred = pd.DataFrame(index=human_activity.index)
    mouse_pred = pd.DataFrame(index=mouse_activity.index)
    calib_rows = []

    for signature, score_name, target_col in specs:
        x = human_activity[signature].to_numpy(dtype=np.float64)
        y = y_human[:, target_col].astype(np.float64)
        if np.std(x) < 1e-12:
            slope = 0.0
            intercept = float(np.mean(y))
        else:
            slope, intercept = np.polyfit(x, y, deg=1)

        human_yhat = intercept + slope * x
        mouse_yhat = intercept + slope * mouse_activity[signature].to_numpy(dtype=np.float64)
        human_pred[f"predicted_signature_{score_name}"] = human_yhat
        mouse_pred[f"predicted_signature_{score_name}"] = mouse_yhat

        calib_rows.append(
            {
                "signature": signature,
                "target_score": score_name,
                "calibration_slope": float(slope),
                "calibration_intercept": float(intercept),
                "human_train_rmse": float(mean_squared_error(y, human_yhat) ** 0.5),
                "human_train_r2": float(r2_score(y, human_yhat)),
            }
        )

    return human_pred, mouse_pred, pd.DataFrame(calib_rows)


def score_frame(sample_ids, predictions, prefix):
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            f"predicted_{prefix}_nas_score": predictions[:, 0],
            f"predicted_{prefix}_fibrosis_stage": predictions[:, 1],
        }
    )


def signature_score_frame(sample_ids, activity, predictions):
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "signature_mas_activity": activity["MAS"].to_numpy(),
            "signature_fibrosis_stage_activity": activity["Fibrosis stage"].to_numpy(),
            "predicted_signature_nas_score": predictions["predicted_signature_nas_score"].to_numpy(),
            "predicted_signature_fibrosis_stage": predictions[
                "predicted_signature_fibrosis_stage"
            ].to_numpy(),
        }
    )


def write_outputs(args, out_dir):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")

    out_dir.mkdir(parents=True, exist_ok=True)

    human_X = np.load(require_file(args.preproc_dir / "human_test_X.npy"), mmap_mode="r")
    mouse_X = np.load(require_file(args.preproc_dir / "mouse_test_X.npy"), mmap_mode="r")
    human_genes = np.load(require_file(args.preproc_dir / "human_genes.npy"), allow_pickle=True)
    human_ids = np.load(require_file(args.preproc_dir / "human_test_sample_ids.npy"), allow_pickle=True)
    mouse_ids = np.load(require_file(args.preproc_dir / "mouse_test_sample_ids.npy"), allow_pickle=True)

    human_meta = read_metadata(args.splits_dir / "liver_metadata_human.csv")
    mouse_meta = read_metadata(args.splits_dir / "liver_metadata_mouse.csv")

    human_subset = select_human_govaere(human_meta, human_ids, args.human_gse)
    mouse_subset = select_mouse_nlrp3(mouse_meta, mouse_ids, args.mouse_gse)

    human_rows = rows_for_ids(human_subset.index.tolist(), human_ids, "human")
    mouse_rows = rows_for_ids(mouse_subset.index.tolist(), mouse_ids, "mouse")

    X_human = load_rows(human_X, human_rows)
    y_human = human_subset[["nas_score", "fibrosis_stage"]].to_numpy(dtype=np.float32)
    human_center = X_human.mean(axis=0, dtype=np.float64).astype(np.float32)
    X_human_centered = X_human - human_center

    pls, used_components = fit_plsr_scorer(
        X_human_centered, y_human, n_components=args.pls_components
    )
    human_pred = pls.predict(X_human_centered)

    enc_m, flow_m2h, dec_h = load_flowtransop_m2h(
        args.fold, human_X.shape[1], mouse_X.shape[1], args.model_dir, device
    )
    translated_mouse = translate_mouse_to_human(
        mouse_X, mouse_rows, enc_m, flow_m2h, dec_h, device, args.batch_size
    )
    translated_mouse_centered = translated_mouse - human_center
    mouse_pred = pls.predict(translated_mouse_centered)

    human_activity, signature_targets = signed_rank_signature_activity(X_human_centered, human_genes)
    mouse_activity, _ = signed_rank_signature_activity(translated_mouse_centered, human_genes)
    human_signature_pred, mouse_signature_pred, signature_calibration = calibrate_signature_predictions(
        human_activity, mouse_activity, y_human
    )

    human_out = score_frame(human_subset.index.to_numpy(), human_pred, "plsr")
    human_out.insert(1, "observed_nas_score", human_subset["nas_score"].to_numpy())
    human_out.insert(2, "observed_fibrosis_stage", human_subset["fibrosis_stage"].to_numpy())
    human_out = human_out.merge(
        human_subset[["characteristics_ch1", "source_name_ch1", "title"]],
        left_on="sample_id",
        right_index=True,
        how="left",
    )

    mouse_out = score_frame(mouse_subset.index.to_numpy(), mouse_pred, "translated_human_plsr")
    mouse_out.insert(1, "mouse_genotype", mouse_subset["mouse_genotype"].to_numpy())
    mouse_out.insert(2, "mouse_treatment", mouse_subset["mouse_treatment"].to_numpy())
    mouse_out = mouse_out.merge(
        mouse_subset[["characteristics_ch1", "source_name_ch1", "title"]],
        left_on="sample_id",
        right_index=True,
        how="left",
    )

    human_sig_out = signature_score_frame(
        human_subset.index.to_numpy(), human_activity, human_signature_pred
    )
    human_sig_out.insert(1, "observed_nas_score", human_subset["nas_score"].to_numpy())
    human_sig_out.insert(2, "observed_fibrosis_stage", human_subset["fibrosis_stage"].to_numpy())
    human_sig_out = human_sig_out.merge(
        human_subset[["characteristics_ch1", "source_name_ch1", "title"]],
        left_on="sample_id",
        right_index=True,
        how="left",
    )

    mouse_sig_out = signature_score_frame(
        mouse_subset.index.to_numpy(), mouse_activity, mouse_signature_pred
    )
    mouse_sig_out.insert(1, "mouse_genotype", mouse_subset["mouse_genotype"].to_numpy())
    mouse_sig_out.insert(2, "mouse_treatment", mouse_subset["mouse_treatment"].to_numpy())
    mouse_sig_out = mouse_sig_out.merge(
        mouse_subset[["characteristics_ch1", "source_name_ch1", "title"]],
        left_on="sample_id",
        right_index=True,
        how="left",
    )

    summary = pd.DataFrame(
        [
            {
                "fold": args.fold,
                "human_gse": args.human_gse,
                "mouse_gse": args.mouse_gse,
                "human_samples": int(len(human_subset)),
                "mouse_samples": int(len(mouse_subset)),
                "human_features": int(human_X.shape[1]),
                "mouse_features": int(mouse_X.shape[1]),
                "pls_components_requested": int(args.pls_components),
                "pls_components_used": int(used_components),
                "human_train_rmse_nas": float(mean_squared_error(y_human[:, 0], human_pred[:, 0]) ** 0.5),
                "human_train_rmse_fibrosis": float(mean_squared_error(y_human[:, 1], human_pred[:, 1]) ** 0.5),
                "human_train_r2_nas": float(r2_score(y_human[:, 0], human_pred[:, 0])),
                "human_train_r2_fibrosis": float(r2_score(y_human[:, 1], human_pred[:, 1])),
            }
        ]
    )
    signature_summary = signature_targets.merge(
        signature_calibration, left_on="signature", right_on="signature", how="left"
    )

    suffix = f"fold{args.fold}"
    human_path = out_dir / f"human_govaere_plsr_scores_{suffix}.csv"
    mouse_path = out_dir / f"mouse_nlrp3_translated_plsr_scores_{suffix}.csv"
    summary_path = out_dir / f"plsr_summary_{suffix}.csv"
    human_signature_path = out_dir / f"human_govaere_signature_scores_{suffix}.csv"
    mouse_signature_path = out_dir / f"mouse_nlrp3_translated_signature_scores_{suffix}.csv"
    signature_summary_path = out_dir / f"signature_summary_{suffix}.csv"
    signature_regulon_path = out_dir / "signature_regulon.csv"

    human_out.to_csv(human_path, index=False)
    mouse_out.to_csv(mouse_path, index=False)
    summary.to_csv(summary_path, index=False)
    human_sig_out.to_csv(human_signature_path, index=False)
    mouse_sig_out.to_csv(mouse_signature_path, index=False)
    signature_summary.to_csv(signature_summary_path, index=False)
    signature_regulon_frame().to_csv(signature_regulon_path, index=False)

    print(f"Wrote human Govaere scores: {human_path}")
    print(f"Wrote translated mouse scores: {mouse_path}")
    print(f"Wrote PLSR summary: {summary_path}")
    print(f"Wrote human signature scores: {human_signature_path}")
    print(f"Wrote translated mouse signature scores: {mouse_signature_path}")
    print(f"Wrote signature summary: {signature_summary_path}")
    print(f"Wrote signature regulon: {signature_regulon_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--splits_dir", type=Path, default=None)
    parser.add_argument("--preproc_dir", type=Path, default=None)
    parser.add_argument("--model_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--human_gse", default=HUMAN_GSE)
    parser.add_argument("--mouse_gse", default=MOUSE_GSE)
    parser.add_argument("--pls_components", type=int, default=PLS_COMPONENTS)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.splits_dir = args.splits_dir or args.data_dir / "splits"
    args.preproc_dir = args.preproc_dir or args.data_dir / "preprocessed"
    args.model_dir = args.model_dir or args.data_dir / "models"
    args.out_dir = args.out_dir or args.data_dir / "evaluation" / "liver_mas_fibrosis"
    return args


def main():
    args = parse_args()
    write_outputs(args, args.out_dir)


if __name__ == "__main__":
    main()
