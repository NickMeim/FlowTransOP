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
4. Fits an additional orthologue-only human PLSR/aREA scorer and applies it
   directly to untranslated mouse orthologue expression.
5. Writes all-gene and orthologue-only human PLSR LOOCV predictions once.

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
MOUSE_CDAA_GSE = "GSE269493"
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
    selected["dataset"] = "GSE196908_Nlrp3A350V_GS444217"
    selected["mouse_model"] = "Nlrp3A350V"
    selected["time_point"] = "not_specified"
    return selected


def _matrix_limited_mouse_meta(meta, mouse_test_ids):
    ids_in_matrix = pd.Index(mouse_test_ids.astype(str))
    return meta.loc[meta.index.intersection(ids_in_matrix)].copy()


def _contains(series, pattern, regex=False):
    return series.astype(str).str.contains(pattern, case=False, regex=regex, na=False)


def select_mouse_gan_dio_lanifibranor(meta, mouse_test_ids, accession=MOUSE_GSE):
    meta = _matrix_limited_mouse_meta(meta, mouse_test_ids)
    chars = meta["characteristics_ch1"].astype(str)

    gse = series_mask(meta, accession)
    if not gse.any():
        gse = pd.Series(True, index=meta.index)

    vehicle_12 = _contains(chars, "DIO-NASH Vehicle 12w")
    vehicle_8 = _contains(chars, "DIO-NASH Vehicle 8w")
    lani_12 = _contains(chars, "Lanifibranor 12w")
    lani_8 = _contains(chars, "Lanifibranor 8w")

    selected = meta.loc[gse & (vehicle_12 | vehicle_8 | lani_12 | lani_8)].copy()
    if selected.empty:
        raise ValueError(
            "No GAN DIO-NASH Lanifibranor samples found for DIO-NASH Vehicle "
            "8w/12w or Lanifibranor 8w/12w."
        )

    sel_chars = selected["characteristics_ch1"].astype(str)
    selected["dataset"] = "GSE196908_GAN_DIO_NASH_Lanifibranor"
    selected["mouse_model"] = "GAN DIO-NASH"
    selected["mouse_treatment"] = np.where(
        _contains(sel_chars, "Lanifibranor"),
        "Lanifibranor",
        "DIO-NASH Vehicle",
    )
    selected["time_point"] = np.where(_contains(sel_chars, "12w"), "12w", "8w")
    return selected


def select_mouse_cdaa_lanifibranor(meta, mouse_test_ids, accession=MOUSE_CDAA_GSE):
    meta = _matrix_limited_mouse_meta(meta, mouse_test_ids)
    chars = meta["characteristics_ch1"].astype(str)

    gse = series_mask(meta, accession)
    vehicle = _contains(chars, "CDAA-HFD vehicle")
    lanifibranor = _contains(chars, "CDAA-HFD Lanifibranor")
    chow = _contains(chars, r"tissue:\s*Liver\s*,\s*treatment:\s*Chow\s*$", regex=True)

    if gse.any():
        mask = gse & (vehicle | lanifibranor | chow)
    else:
        disease_protocols = meta.loc[vehicle | lanifibranor, "extract_protocol_ch1"].astype(str)
        if disease_protocols.empty:
            raise ValueError(
                "No CDAA-HFD vehicle or CDAA-HFD Lanifibranor samples found "
                "to anchor the GSE269493 selector."
            )
        shared_protocol = disease_protocols.mode().iloc[0]
        same_protocol = meta["extract_protocol_ch1"].astype(str) == shared_protocol
        mask = vehicle | lanifibranor | (chow & same_protocol)

    selected = meta.loc[mask].copy()
    if selected.empty:
        raise ValueError(
            "No GSE269493 CDAA-HFD vehicle/Lanifibranor/Chow samples found."
        )

    sel_chars = selected["characteristics_ch1"].astype(str)
    selected["dataset"] = "GSE269493_CDAA_HFD_Lanifibranor"
    selected["mouse_model"] = np.where(_contains(sel_chars, "CDAA-HFD"), "CDAA-HFD", "Healthy")
    selected["mouse_treatment"] = np.select(
        [
            _contains(sel_chars, "CDAA-HFD Lanifibranor"),
            _contains(sel_chars, "CDAA-HFD vehicle"),
            _contains(sel_chars, r"treatment:\s*Chow", regex=True),
        ],
        ["Lanifibranor", "CDAA-HFD vehicle", "Chow"],
        default="unknown",
    )
    selected["time_point"] = "not_specified"
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


def fit_signature_calibration(human_activity, y_human):
    specs = [
        ("MAS", "nas_score", 0),
        ("Fibrosis stage", "fibrosis_stage", 1),
    ]
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

    return pd.DataFrame(calib_rows)


def calibrate_signature_predictions(human_activity, mouse_activity, y_human):
    calibration = fit_signature_calibration(human_activity, y_human)
    human_pred = apply_signature_calibration(human_activity, calibration)
    mouse_pred = apply_signature_calibration(mouse_activity, calibration)
    return human_pred, mouse_pred, calibration


def apply_signature_calibration(activity, calibration):
    predictions = pd.DataFrame(index=activity.index)
    for _, row in calibration.iterrows():
        signature = row["signature"]
        target_score = row["target_score"]
        predictions[f"predicted_signature_{target_score}"] = (
            float(row["calibration_intercept"])
            + float(row["calibration_slope"]) * activity[signature].to_numpy(dtype=np.float64)
        )
    return predictions


def loocv_plsr_predictions(X, y, sample_ids, n_components, gene_space):
    rows = []
    sample_ids = np.asarray(sample_ids).astype(str)
    for held_out in range(X.shape[0]):
        train_mask = np.ones(X.shape[0], dtype=bool)
        train_mask[held_out] = False
        X_train = X[train_mask]
        y_train = y[train_mask]
        center = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        X_train_centered = X_train - center
        model, used_components = fit_plsr_scorer(
            X_train_centered, y_train, n_components=n_components
        )
        pred = model.predict(X[held_out:held_out + 1] - center)[0]
        rows.append(
            {
                "sample_id": sample_ids[held_out],
                "gene_space": gene_space,
                "observed_nas_score": float(y[held_out, 0]),
                "observed_fibrosis_stage": float(y[held_out, 1]),
                "predicted_loocv_nas_score": float(pred[0]),
                "predicted_loocv_fibrosis_stage": float(pred[1]),
                "n_train_samples": int(train_mask.sum()),
                "n_features": int(X.shape[1]),
                "pls_components_requested": int(n_components),
                "pls_components_used": int(used_components),
            }
        )
    return pd.DataFrame(rows)


def score_frame(sample_ids, predictions, prefix):
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            f"predicted_{prefix}_nas_score": predictions[:, 0],
            f"predicted_{prefix}_fibrosis_stage": predictions[:, 1],
        }
    )


def attach_mouse_metadata(score_df, mouse_subset):
    optional_cols = [
        "dataset",
        "mouse_model",
        "mouse_genotype",
        "mouse_treatment",
        "time_point",
        "characteristics_ch1",
        "source_name_ch1",
        "title",
    ]
    cols = [col for col in optional_cols if col in mouse_subset.columns]
    return score_df.merge(
        mouse_subset[cols],
        left_on="sample_id",
        right_index=True,
        how="left",
    )


def write_loocv_if_needed(args, out_dir, X_human, X_human_orth, y_human, human_ids):
    if args.skip_loocv or args.fold != args.loocv_write_fold:
        return []
    all_genes = loocv_plsr_predictions(
        X_human, y_human, human_ids, args.pls_components, "all_genes"
    )
    orthologues = loocv_plsr_predictions(
        X_human_orth, y_human, human_ids, args.pls_components, "orthologues"
    )
    all_path = out_dir / "human_govaere_plsr_loocv_all_genes.csv"
    orth_path = out_dir / "human_govaere_plsr_loocv_orthologues.csv"
    all_genes.to_csv(all_path, index=False)
    orthologues.to_csv(orth_path, index=False)
    return [all_path, orth_path]


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


def score_translated_mouse_subset(
    mouse_subset,
    mouse_X,
    mouse_ids,
    enc_m,
    flow_m2h,
    dec_h,
    human_center,
    pls,
    signature_calibration,
    human_genes,
    device,
    batch_size,
):
    mouse_rows = rows_for_ids(mouse_subset.index.tolist(), mouse_ids, "mouse")
    translated = translate_mouse_to_human(
        mouse_X, mouse_rows, enc_m, flow_m2h, dec_h, device, batch_size
    )
    centered = translated - human_center

    plsr_pred = pls.predict(centered)
    plsr_out = score_frame(mouse_subset.index.to_numpy(), plsr_pred, "translated_human_plsr")
    plsr_out.insert(1, "input_space", "translated_human")
    plsr_out.insert(2, "gene_space", "all_human_genes")
    plsr_out = attach_mouse_metadata(plsr_out, mouse_subset)

    activity, _ = signed_rank_signature_activity(centered, human_genes)
    signature_pred = apply_signature_calibration(activity, signature_calibration)
    sig_out = signature_score_frame(mouse_subset.index.to_numpy(), activity, signature_pred)
    sig_out.insert(1, "input_space", "translated_human")
    sig_out.insert(2, "gene_space", "all_human_genes")
    sig_out = attach_mouse_metadata(sig_out, mouse_subset)

    return plsr_out, sig_out


def score_raw_mouse_orthologue_subset(
    mouse_subset,
    mouse_X,
    mouse_ids,
    mouse_orth_idx,
    human_orth_genes,
    human_orth_center,
    pls_orth,
    orth_signature_calibration,
):
    mouse_rows = rows_for_ids(mouse_subset.index.tolist(), mouse_ids, "mouse")
    X_mouse_orth = load_rows(mouse_X, mouse_rows)[:, mouse_orth_idx]
    X_mouse_orth_centered = X_mouse_orth - human_orth_center

    plsr_pred = pls_orth.predict(X_mouse_orth_centered)
    plsr_out = score_frame(mouse_subset.index.to_numpy(), plsr_pred, "raw_mouse_orthologue_plsr")
    plsr_out.insert(1, "input_space", "raw_mouse_orthologues")
    plsr_out.insert(2, "gene_space", "orthologues")
    plsr_out = attach_mouse_metadata(plsr_out, mouse_subset)

    activity, _ = signed_rank_signature_activity(X_mouse_orth, human_orth_genes)
    signature_pred = apply_signature_calibration(activity, orth_signature_calibration)
    sig_out = signature_score_frame(mouse_subset.index.to_numpy(), activity, signature_pred)
    sig_out.insert(1, "input_space", "raw_mouse_orthologues")
    sig_out.insert(2, "gene_space", "orthologues")
    sig_out = attach_mouse_metadata(sig_out, mouse_subset)

    return plsr_out, sig_out


def write_outputs(args, out_dir):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")

    out_dir.mkdir(parents=True, exist_ok=True)

    human_X = np.load(require_file(args.preproc_dir / "human_test_X.npy"), mmap_mode="r")
    mouse_X = np.load(require_file(args.preproc_dir / "mouse_test_X.npy"), mmap_mode="r")
    human_genes = np.load(require_file(args.preproc_dir / "human_genes.npy"), allow_pickle=True)
    mouse_genes = np.load(require_file(args.preproc_dir / "mouse_genes.npy"), allow_pickle=True)
    human_ids = np.load(require_file(args.preproc_dir / "human_test_sample_ids.npy"), allow_pickle=True)
    mouse_ids = np.load(require_file(args.preproc_dir / "mouse_test_sample_ids.npy"), allow_pickle=True)

    human_meta = read_metadata(args.splits_dir / "liver_metadata_human.csv")
    mouse_meta = read_metadata(args.splits_dir / "liver_metadata_mouse.csv")

    human_subset = select_human_govaere(human_meta, human_ids, args.human_gse)
    nlrp3_subset = select_mouse_nlrp3(mouse_meta, mouse_ids, args.mouse_gse)
    gan_lani_subset = select_mouse_gan_dio_lanifibranor(mouse_meta, mouse_ids, args.mouse_gse)
    cdaa_lani_subset = select_mouse_cdaa_lanifibranor(mouse_meta, mouse_ids, args.mouse_cdaa_gse)
    mouse_subsets = {
        "mouse_nlrp3": nlrp3_subset,
        "mouse_gan_dio_lanifibranor": gan_lani_subset,
        "mouse_cdaa_lanifibranor": cdaa_lani_subset,
    }

    human_rows = rows_for_ids(human_subset.index.tolist(), human_ids, "human")

    X_human = load_rows(human_X, human_rows)
    y_human = human_subset[["nas_score", "fibrosis_stage"]].to_numpy(dtype=np.float32)
    human_center = X_human.mean(axis=0, dtype=np.float64).astype(np.float32)
    X_human_centered = X_human - human_center

    human_orth_idx, mouse_orth_idx = et.build_orthologue_map(human_genes, mouse_genes)
    human_orth_genes = human_genes[human_orth_idx]
    X_human_orth = X_human[:, human_orth_idx]
    human_orth_center = X_human_orth.mean(axis=0, dtype=np.float64).astype(np.float32)
    X_human_orth_centered = X_human_orth - human_orth_center

    pls, used_components = fit_plsr_scorer(
        X_human_centered, y_human, n_components=args.pls_components
    )
    human_pred = pls.predict(X_human_centered)

    pls_orth, used_orth_components = fit_plsr_scorer(
        X_human_orth_centered, y_human, n_components=args.pls_components
    )
    human_orth_pred = pls_orth.predict(X_human_orth_centered)

    loocv_paths = write_loocv_if_needed(
        args,
        out_dir,
        X_human,
        X_human_orth,
        y_human,
        human_subset.index.to_numpy(),
    )

    enc_m, flow_m2h, dec_h = load_flowtransop_m2h(
        args.fold, human_X.shape[1], mouse_X.shape[1], args.model_dir, device
    )

    human_activity, signature_targets = signed_rank_signature_activity(X_human_centered, human_genes)
    signature_calibration = fit_signature_calibration(human_activity, y_human)
    human_signature_pred = apply_signature_calibration(human_activity, signature_calibration)

    human_orth_activity, orth_signature_targets = signed_rank_signature_activity(
        X_human_orth_centered, human_orth_genes
    )
    orth_signature_calibration = fit_signature_calibration(human_orth_activity, y_human)
    human_orth_signature_pred = apply_signature_calibration(
        human_orth_activity, orth_signature_calibration
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

    human_orth_out = score_frame(human_subset.index.to_numpy(), human_orth_pred, "orthologue_plsr")
    human_orth_out.insert(1, "observed_nas_score", human_subset["nas_score"].to_numpy())
    human_orth_out.insert(2, "observed_fibrosis_stage", human_subset["fibrosis_stage"].to_numpy())
    human_orth_out.insert(3, "gene_space", "orthologues")
    human_orth_out = human_orth_out.merge(
        human_subset[["characteristics_ch1", "source_name_ch1", "title"]],
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

    human_orth_sig_out = signature_score_frame(
        human_subset.index.to_numpy(), human_orth_activity, human_orth_signature_pred
    )
    human_orth_sig_out.insert(1, "observed_nas_score", human_subset["nas_score"].to_numpy())
    human_orth_sig_out.insert(2, "observed_fibrosis_stage", human_subset["fibrosis_stage"].to_numpy())
    human_orth_sig_out.insert(3, "gene_space", "orthologues")
    human_orth_sig_out = human_orth_sig_out.merge(
        human_subset[["characteristics_ch1", "source_name_ch1", "title"]],
        left_on="sample_id",
        right_index=True,
        how="left",
    )

    translated_outputs = {}
    raw_orth_outputs = {}
    for cohort_name, subset in mouse_subsets.items():
        translated_outputs[cohort_name] = score_translated_mouse_subset(
            subset,
            mouse_X,
            mouse_ids,
            enc_m,
            flow_m2h,
            dec_h,
            human_center,
            pls,
            signature_calibration,
            human_genes,
            device,
            args.batch_size,
        )
        raw_orth_outputs[cohort_name] = score_raw_mouse_orthologue_subset(
            subset,
            mouse_X,
            mouse_ids,
            mouse_orth_idx,
            human_orth_genes,
            human_orth_center,
            pls_orth,
            orth_signature_calibration,
        )

    summary = pd.DataFrame(
        [
            {
                "fold": args.fold,
                "human_gse": args.human_gse,
                "mouse_gse": args.mouse_gse,
                "mouse_cdaa_gse": args.mouse_cdaa_gse,
                "human_samples": int(len(human_subset)),
                "mouse_nlrp3_samples": int(len(nlrp3_subset)),
                "mouse_gan_dio_lanifibranor_samples": int(len(gan_lani_subset)),
                "mouse_cdaa_lanifibranor_samples": int(len(cdaa_lani_subset)),
                "human_features": int(human_X.shape[1]),
                "mouse_features": int(mouse_X.shape[1]),
                "orthologue_features": int(len(human_orth_idx)),
                "pls_components_requested": int(args.pls_components),
                "pls_components_used": int(used_components),
                "orthologue_pls_components_used": int(used_orth_components),
                "human_train_rmse_nas": float(mean_squared_error(y_human[:, 0], human_pred[:, 0]) ** 0.5),
                "human_train_rmse_fibrosis": float(mean_squared_error(y_human[:, 1], human_pred[:, 1]) ** 0.5),
                "human_train_r2_nas": float(r2_score(y_human[:, 0], human_pred[:, 0])),
                "human_train_r2_fibrosis": float(r2_score(y_human[:, 1], human_pred[:, 1])),
                "human_orthologue_train_rmse_nas": float(mean_squared_error(y_human[:, 0], human_orth_pred[:, 0]) ** 0.5),
                "human_orthologue_train_rmse_fibrosis": float(mean_squared_error(y_human[:, 1], human_orth_pred[:, 1]) ** 0.5),
                "human_orthologue_train_r2_nas": float(r2_score(y_human[:, 0], human_orth_pred[:, 0])),
                "human_orthologue_train_r2_fibrosis": float(r2_score(y_human[:, 1], human_orth_pred[:, 1])),
            }
        ]
    )
    signature_summary = signature_targets.merge(
        signature_calibration, left_on="signature", right_on="signature", how="left"
    )
    signature_summary.insert(1, "gene_space", "all_human_genes")
    orth_signature_summary = orth_signature_targets.merge(
        orth_signature_calibration, left_on="signature", right_on="signature", how="left"
    )
    orth_signature_summary.insert(1, "gene_space", "orthologues")
    signature_summary = pd.concat(
        [signature_summary, orth_signature_summary],
        ignore_index=True,
    )

    suffix = f"fold{args.fold}"
    human_path = out_dir / f"human_govaere_plsr_scores_{suffix}.csv"
    human_orth_path = out_dir / f"human_govaere_orthologue_plsr_scores_{suffix}.csv"
    summary_path = out_dir / f"plsr_summary_{suffix}.csv"
    human_signature_path = out_dir / f"human_govaere_signature_scores_{suffix}.csv"
    human_orth_signature_path = out_dir / f"human_govaere_orthologue_signature_scores_{suffix}.csv"
    signature_summary_path = out_dir / f"signature_summary_{suffix}.csv"
    signature_regulon_path = out_dir / "signature_regulon.csv"
    orthologue_map_path = out_dir / "orthologue_gene_map.csv"

    human_out.to_csv(human_path, index=False)
    human_orth_out.to_csv(human_orth_path, index=False)
    summary.to_csv(summary_path, index=False)
    human_sig_out.to_csv(human_signature_path, index=False)
    human_orth_sig_out.to_csv(human_orth_signature_path, index=False)
    signature_summary.to_csv(signature_summary_path, index=False)
    signature_regulon_frame().to_csv(signature_regulon_path, index=False)
    pd.DataFrame(
        {
            "human_gene": human_genes[human_orth_idx].astype(str),
            "mouse_gene": mouse_genes[mouse_orth_idx].astype(str),
            "human_gene_index": human_orth_idx,
            "mouse_gene_index": mouse_orth_idx,
        }
    ).to_csv(orthologue_map_path, index=False)

    output_paths = [
        human_path,
        human_orth_path,
        summary_path,
        human_signature_path,
        human_orth_signature_path,
        signature_summary_path,
        signature_regulon_path,
        orthologue_map_path,
    ]
    output_paths.extend(loocv_paths)

    for cohort_name, (plsr_out, sig_out) in translated_outputs.items():
        plsr_path = out_dir / f"{cohort_name}_translated_plsr_scores_{suffix}.csv"
        sig_path = out_dir / f"{cohort_name}_translated_signature_scores_{suffix}.csv"
        plsr_out.to_csv(plsr_path, index=False)
        sig_out.to_csv(sig_path, index=False)
        output_paths.extend([plsr_path, sig_path])

    for cohort_name, (plsr_out, sig_out) in raw_orth_outputs.items():
        plsr_path = out_dir / f"{cohort_name}_raw_orthologue_plsr_scores_{suffix}.csv"
        sig_path = out_dir / f"{cohort_name}_raw_orthologue_signature_scores_{suffix}.csv"
        plsr_out.to_csv(plsr_path, index=False)
        sig_out.to_csv(sig_path, index=False)
        output_paths.extend([plsr_path, sig_path])

    for path in output_paths:
        print(f"Wrote: {path}")


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
    parser.add_argument("--mouse_cdaa_gse", default=MOUSE_CDAA_GSE)
    parser.add_argument("--pls_components", type=int, default=PLS_COMPONENTS)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--skip_loocv",
        action="store_true",
        help="Skip human PLSR leave-one-out predictions.",
    )
    parser.add_argument(
        "--loocv_write_fold",
        type=int,
        default=0,
        help="Only this fold writes the non-fold-specific LOOCV CSVs.",
    )
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
