#!/usr/bin/env python3
"""
Score ARCHS4 liver subsets for NAS/MAS activity and fibrosis.

This script uses the preprocessed liver test matrices written by the ARCHS4
preprocessing scripts and trained FlowTransOP fold or full-ensemble checkpoints.
It:

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
   directly to untranslated mouse orthologue expression and to translated,
   corrected human orthologue expression.
5. Fits PLSR and RBF-SVM scorers in expression and latent spaces and applies
   them to the matching mouse representations.
6. Optionally samples from the translated decoder Gaussian or NB distribution
   to score reconstructed expression distributions instead of only decoder
   means.
7. Writes all-gene and orthologue-only human PLSR LOOCV predictions once,
   plus RBF-SVM LOOCV predictions for expression, orthologue, and latent
   feature spaces.

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
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

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
SVM_RBF_PARAM_GRID = {
    "gamma": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1.5, 2],
    "C": [1e-1, 1, 10, 100, 1000],
}
TARGET_NAMES = ("nas_score", "fibrosis_stage")

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


def active_model_index(args):
    return int(args.fold if args.model_source == "fold" else args.ensemble_id)


def output_suffix(args):
    if args.output_suffix:
        return args.output_suffix
    if args.model_source == "fold" or args.ensemble_suffix_as_fold:
        return f"fold{active_model_index(args)}"
    return f"ensemble{args.ensemble_id}"


def model_checkpoint_paths(args):
    if args.model_source == "fold":
        return (
            args.model_dir / f"fold_{args.fold}_normal.pt",
            args.model_dir / f"fold_{args.fold}_normal_m2h.pt",
        )
    if args.model_source == "full_ensemble":
        stem = f"{args.full_model_prefix}_{args.ensemble_id}"
        return (
            args.model_dir / f"{stem}_normal.pt",
            args.model_dir / f"{stem}_normal_m2h.pt",
        )
    raise ValueError(f"Unknown model_source: {args.model_source}")


def load_flowtransop_m2h(args, human_dim, mouse_dim, device):
    normal_path, m2h_path = model_checkpoint_paths(args)
    normal_path = require_file(normal_path)

    ckpt = torch.load(normal_path, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]

    enc_h, enc_m, dec_h, _, _ = et.build_models(ckpt_args, human_dim, mouse_dim, device)
    enc_h.load_state_dict(ckpt["encoder_human"])
    enc_m.load_state_dict(ckpt["encoder_mouse"])
    dec_h.load_state_dict(ckpt["decoder_human"])

    flow_m2h = et.Flow(
        ckpt_args["latent_dim"], ckpt_args["latent_dim"] // 2, dtype=torch.float
    ).to(device)
    if m2h_path.exists():
        ckpt_m2h = torch.load(m2h_path, map_location=device, weights_only=False)
        flow_m2h.load_state_dict(ckpt_m2h["flow_m2h"])
    elif "flow_m2h" in ckpt:
        flow_m2h.load_state_dict(ckpt["flow_m2h"])
    else:
        raise FileNotFoundError(
            f"Could not find mouse->human flow in {m2h_path} or {normal_path}."
        )

    enc_h.eval()
    enc_m.eval()
    flow_m2h.eval()
    dec_h.eval()
    return enc_h, enc_m, flow_m2h, dec_h


@torch.no_grad()
def encode_human_latent(human_matrix, enc_h, device, batch_size):
    latents = []
    for start in range(0, human_matrix.shape[0], batch_size):
        stop = min(start + batch_size, human_matrix.shape[0])
        block = np.asarray(human_matrix[start:stop, :], dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(block)).to(device)
        latents.append(enc_h(x).cpu().numpy())
    return np.concatenate(latents, axis=0).astype(np.float32)


@torch.no_grad()
def translate_mouse_to_human_distribution_and_latent(
    mouse_matrix,
    mouse_rows,
    enc_m,
    flow_m2h,
    dec_h,
    device,
    batch_size,
):
    out_dim = int(dec_h.out_mu.out_features)
    translated = np.empty((len(mouse_rows), out_dim), dtype=np.float32)
    translated_var = np.empty((len(mouse_rows), out_dim), dtype=np.float32)
    latents = []
    for start in range(0, len(mouse_rows), batch_size):
        stop = min(start + batch_size, len(mouse_rows))
        block = np.asarray(mouse_matrix[mouse_rows[start:stop], :], dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(block)).to(device)
        z = enc_m(x)
        z_h = et.flow_step_n(flow_m2h, z)
        mu, var = dec_h(z_h)
        translated[start:stop] = mu.cpu().numpy()
        translated_var[start:stop] = var.cpu().numpy()
        latents.append(z_h.cpu().numpy())
    translated_latent = np.concatenate(latents, axis=0).astype(np.float32)
    return translated, translated_var, translated_latent


def translate_mouse_to_human_and_latent(
    mouse_matrix,
    mouse_rows,
    enc_m,
    flow_m2h,
    dec_h,
    device,
    batch_size,
):
    translated, _, translated_latent = translate_mouse_to_human_distribution_and_latent(
        mouse_matrix, mouse_rows, enc_m, flow_m2h, dec_h, device, batch_size
    )
    return translated, translated_latent


def translate_mouse_to_human(mouse_matrix, mouse_rows, enc_m, flow_m2h, dec_h, device, batch_size):
    translated, _ = translate_mouse_to_human_and_latent(
        mouse_matrix, mouse_rows, enc_m, flow_m2h, dec_h, device, batch_size
    )
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


def fit_svm_rbf_regressor(X, y, args):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    cv = min(args.svm_cv, X.shape[0])
    if cv < 2:
        raise ValueError("At least two human samples are required for RBF-SVM CV.")
    grid = GridSearchCV(
        estimator=SVR(kernel="rbf"),
        param_grid=SVM_RBF_PARAM_GRID,
        cv=cv,
        n_jobs=args.svm_jobs,
    )
    model = MultiOutputRegressor(grid)
    model.fit(X, y)
    return {"kind": "svm_rbf", "model": model, "cv": int(cv)}


def fit_ml_model_set(X, y, args, device, plsr_model=None, plsr_components=None):
    models = {}
    if plsr_model is None:
        plsr_model, plsr_components = fit_plsr_scorer(
            X, y, n_components=args.pls_components
        )
    models["plsr"] = {
        "kind": "plsr",
        "model": plsr_model,
        "components": int(plsr_components),
    }
    if not args.skip_svm_rbf:
        models["svm_rbf"] = fit_svm_rbf_regressor(X, y, args)
    return models


def predict_ml_model_set(models, X, args, device):
    predictions = {}
    for model_name, bundle in models.items():
        if bundle["kind"] == "plsr":
            pred = bundle["model"].predict(X)
        elif bundle["kind"] == "svm_rbf":
            pred = bundle["model"].predict(np.asarray(X, dtype=np.float32))
        else:
            raise ValueError(f"Unknown model kind: {bundle['kind']}")
        predictions[model_name] = np.asarray(pred, dtype=np.float32)
    return predictions


def ml_prediction_long_frame(
    sample_ids,
    models,
    X,
    args,
    device,
    input_space,
    gene_space,
    observed=None,
):
    frames = []
    predictions = predict_ml_model_set(models, X, args, device)
    for model_name, pred in predictions.items():
        frame = pd.DataFrame(
            {
                "sample_id": np.asarray(sample_ids).astype(str),
                "model_type": model_name,
                "input_space": input_space,
                "gene_space": gene_space,
                "predicted_nas_score": pred[:, 0],
                "predicted_fibrosis_stage": pred[:, 1],
            }
        )
        if observed is not None:
            frame.insert(1, "observed_nas_score", np.asarray(observed)[:, 0])
            frame.insert(2, "observed_fibrosis_stage", np.asarray(observed)[:, 1])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_ml_model_set(models, X, y, args, device, input_space, gene_space):
    rows = []
    predictions = predict_ml_model_set(models, X, args, device)
    for model_name, pred in predictions.items():
        row = {
            "model_type": model_name,
            "input_space": input_space,
            "gene_space": gene_space,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "train_rmse_nas": float(mean_squared_error(y[:, 0], pred[:, 0]) ** 0.5),
            "train_rmse_fibrosis": float(mean_squared_error(y[:, 1], pred[:, 1]) ** 0.5),
            "train_r2_nas": float(r2_score(y[:, 0], pred[:, 0])),
            "train_r2_fibrosis": float(r2_score(y[:, 1], pred[:, 1])),
        }
        bundle = models[model_name]
        if bundle["kind"] == "plsr":
            row["pls_components_used"] = int(bundle["components"])
        elif bundle["kind"] == "svm_rbf":
            estimators = bundle["model"].estimators_
            row["svm_cv"] = int(bundle["cv"])
            row["svm_best_params_nas"] = repr(estimators[0].best_params_)
            row["svm_best_params_fibrosis"] = repr(estimators[1].best_params_)
        rows.append(row)
    return pd.DataFrame(rows)


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


def loocv_ml_predictions(X, y, sample_ids, args, device, input_space, gene_space):
    rows = []
    sample_ids = np.asarray(sample_ids).astype(str)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    for held_out in range(X.shape[0]):
        train_mask = np.ones(X.shape[0], dtype=bool)
        train_mask[held_out] = False
        X_train = X[train_mask]
        y_train = y[train_mask]
        center = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        X_train_centered = X_train - center
        X_test_centered = X[held_out:held_out + 1] - center

        if not args.skip_svm_rbf:
            svm_model = fit_svm_rbf_regressor(X_train_centered, y_train, args)
            svm_pred = svm_model["model"].predict(X_test_centered)[0]
            estimators = svm_model["model"].estimators_
            rows.append(
                {
                    "sample_id": sample_ids[held_out],
                    "model_type": "svm_rbf",
                    "input_space": input_space,
                    "gene_space": gene_space,
                    "observed_nas_score": float(y[held_out, 0]),
                    "observed_fibrosis_stage": float(y[held_out, 1]),
                    "predicted_loocv_nas_score": float(svm_pred[0]),
                    "predicted_loocv_fibrosis_stage": float(svm_pred[1]),
                    "n_train_samples": int(train_mask.sum()),
                    "n_features": int(X.shape[1]),
                    "svm_cv": int(svm_model["cv"]),
                    "svm_best_params_nas": repr(estimators[0].best_params_),
                    "svm_best_params_fibrosis": repr(estimators[1].best_params_),
                }
            )
            del svm_model

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


def write_loocv_if_needed(
    args,
    out_dir,
    X_human,
    X_human_orth,
    y_human,
    human_ids,
    Z_human=None,
    device=None,
):
    if args.skip_loocv or active_model_index(args) != args.loocv_write_fold:
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
    output_paths = [all_path, orth_path]

    if not args.skip_ml_loocv:
        if Z_human is None or device is None:
            raise ValueError("Z_human and device are required for ML LOOCV.")
        ml_loocv = pd.concat(
            [
                loocv_ml_predictions(
                    X_human,
                    y_human,
                    human_ids,
                    args,
                    device,
                    input_space="human_expression",
                    gene_space="all_human_genes",
                ),
                loocv_ml_predictions(
                    X_human_orth,
                    y_human,
                    human_ids,
                    args,
                    device,
                    input_space="human_orthologue_expression",
                    gene_space="orthologues",
                ),
                loocv_ml_predictions(
                    Z_human,
                    y_human,
                    human_ids,
                    args,
                    device,
                    input_space="human_latent",
                    gene_space="latent",
                ),
            ],
            ignore_index=True,
        )
        if not ml_loocv.empty:
            ml_path = out_dir / "human_govaere_ml_loocv_predictions.csv"
            ml_loocv.to_csv(ml_path, index=False)
            output_paths.append(ml_path)

    return output_paths


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


def _stable_name_offset(name):
    return sum((i + 1) * ord(ch) for i, ch in enumerate(str(name)))


def score_prediction_frame(sample_ids, predictions, model_type, input_space, gene_space, decoder_sample=None):
    out = pd.DataFrame(
        {
            "sample_id": np.asarray(sample_ids).astype(str),
            "model_type": model_type,
            "input_space": input_space,
            "gene_space": gene_space,
            "predicted_nas_score": np.asarray(predictions)[:, 0],
            "predicted_fibrosis_stage": np.asarray(predictions)[:, 1],
        }
    )
    if decoder_sample is not None:
        out.insert(1, "decoder_sample", int(decoder_sample))
    return out


def signature_prediction_frame(
    sample_ids,
    activity,
    predictions,
    input_space,
    gene_space,
    decoder_sample=None,
):
    pred = np.column_stack(
        [
            predictions["predicted_signature_nas_score"].to_numpy(dtype=np.float32),
            predictions["predicted_signature_fibrosis_stage"].to_numpy(dtype=np.float32),
        ]
    )
    out = score_prediction_frame(
        sample_ids,
        pred,
        "rank_area",
        input_space,
        gene_space,
        decoder_sample=decoder_sample,
    )
    out["signature_mas_activity"] = activity["MAS"].to_numpy(dtype=np.float32)
    out["signature_fibrosis_stage_activity"] = activity["Fibrosis stage"].to_numpy(dtype=np.float32)
    return out


def decoder_sample_rng(args, cohort_name):
    seed = int(args.decoder_sample_seed)
    seed += 1000003 * active_model_index(args)
    seed += _stable_name_offset(cohort_name)
    return np.random.default_rng(seed)


def summarize_decoder_sample_scores(draws):
    if draws.empty:
        return draws.copy()

    group_cols = ["sample_id", "model_type", "input_space", "gene_space"]
    rows = []
    for keys, group in draws.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        row["n_decoder_samples"] = int(group["decoder_sample"].nunique())
        for score_col, prefix in [
            ("predicted_nas_score", "predicted_nas_score"),
            ("predicted_fibrosis_stage", "predicted_fibrosis_stage"),
        ]:
            values = group[score_col].to_numpy(dtype=np.float64)
            row[f"{prefix}_mean"] = float(np.mean(values))
            row[f"{prefix}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{prefix}_q025"] = float(np.quantile(values, 0.025))
            row[f"{prefix}_q50"] = float(np.quantile(values, 0.5))
            row[f"{prefix}_q975"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def score_translated_orthologue_from_matrix(
    mouse_subset,
    translated,
    human_orth_idx,
    human_orth_genes,
    human_orth_center,
    pls_orth,
    orth_signature_calibration,
):
    translated_orth = translated[:, human_orth_idx]
    translated_orth_centered = translated_orth - human_orth_center

    plsr_pred = pls_orth.predict(translated_orth_centered)
    plsr_out = score_frame(mouse_subset.index.to_numpy(), plsr_pred, "translated_orthologue_plsr")
    plsr_out.insert(1, "input_space", "translated_orthologues")
    plsr_out.insert(2, "gene_space", "orthologues")
    plsr_out = attach_mouse_metadata(plsr_out, mouse_subset)

    activity, _ = signed_rank_signature_activity(translated_orth_centered, human_orth_genes)
    signature_pred = apply_signature_calibration(activity, orth_signature_calibration)
    sig_out = signature_score_frame(mouse_subset.index.to_numpy(), activity, signature_pred)
    sig_out.insert(1, "input_space", "translated_orthologues")
    sig_out.insert(2, "gene_space", "orthologues")
    sig_out = attach_mouse_metadata(sig_out, mouse_subset)

    return plsr_out, sig_out


def score_decoder_sampled_translations(
    mouse_subset,
    translated_mu,
    translated_var,
    human_center,
    human_genes,
    human_orth_idx,
    human_orth_genes,
    human_orth_center,
    expression_ml_models,
    orthologue_ml_models,
    signature_calibration,
    orth_signature_calibration,
    args,
    device,
    cohort_name,
):
    if args.decoder_sample_n <= 0:
        return pd.DataFrame(), pd.DataFrame()

    sample_ids = mouse_subset.index.to_numpy()
    rng = decoder_sample_rng(args, cohort_name)
    frames = []

    for draw_idx in range(args.decoder_sample_n):
        sampled = sample_decoder_expression(translated_mu, translated_var, rng, args)

        centered = sampled - human_center
        all_gene_predictions = predict_ml_model_set(expression_ml_models, centered, args, device)
        for model_type, pred in all_gene_predictions.items():
            frames.append(
                score_prediction_frame(
                    sample_ids,
                    pred,
                    model_type,
                    "decoder_sampled_translated_human",
                    "all_human_genes",
                    decoder_sample=draw_idx,
                )
            )
        activity, _ = signed_rank_signature_activity(centered, human_genes)
        signature_pred = apply_signature_calibration(activity, signature_calibration)
        frames.append(
            signature_prediction_frame(
                sample_ids,
                activity,
                signature_pred,
                "decoder_sampled_translated_human",
                "all_human_genes",
                decoder_sample=draw_idx,
            )
        )

        sampled_orth = sampled[:, human_orth_idx]
        sampled_orth_centered = sampled_orth - human_orth_center
        orth_predictions = predict_ml_model_set(
            orthologue_ml_models, sampled_orth_centered, args, device
        )
        for model_type, pred in orth_predictions.items():
            frames.append(
                score_prediction_frame(
                    sample_ids,
                    pred,
                    model_type,
                    "decoder_sampled_translated_orthologues",
                    "orthologues",
                    decoder_sample=draw_idx,
                )
            )
        orth_activity, _ = signed_rank_signature_activity(sampled_orth_centered, human_orth_genes)
        orth_signature_pred = apply_signature_calibration(orth_activity, orth_signature_calibration)
        frames.append(
            signature_prediction_frame(
                sample_ids,
                orth_activity,
                orth_signature_pred,
                "decoder_sampled_translated_orthologues",
                "orthologues",
                decoder_sample=draw_idx,
            )
        )

    draws = pd.concat(frames, ignore_index=True)
    draws["decoder_sample_n"] = int(args.decoder_sample_n)
    draws["decoder_sample_distribution"] = args.decoder_sample_distribution
    draws["decoder_sample_temperature"] = float(args.decoder_sample_temperature)
    draws["decoder_sample_var_floor"] = float(args.decoder_sample_var_floor)
    draws["decoder_sample_clip_min"] = args.decoder_sample_clip_min
    summary = summarize_decoder_sample_scores(draws)
    summary["decoder_sample_n"] = int(args.decoder_sample_n)
    summary["decoder_sample_distribution"] = args.decoder_sample_distribution
    summary["decoder_sample_temperature"] = float(args.decoder_sample_temperature)
    summary["decoder_sample_var_floor"] = float(args.decoder_sample_var_floor)
    summary["decoder_sample_clip_min"] = args.decoder_sample_clip_min

    draws = attach_mouse_metadata(draws, mouse_subset)
    summary = attach_mouse_metadata(summary, mouse_subset)
    return draws, summary


def sample_decoder_expression(translated_mu, translated_var, rng, args):
    if args.decoder_sample_distribution == "gaussian":
        sd = np.sqrt(np.maximum(translated_var, args.decoder_sample_var_floor)).astype(np.float32)
        noise = rng.normal(loc=0.0, scale=1.0, size=translated_mu.shape).astype(np.float32)
        temperature = max(float(args.decoder_sample_temperature), 0.0)
        sampled = translated_mu + temperature * sd * noise
    elif args.decoder_sample_distribution == "negative_binomial":
        mu = np.maximum(translated_mu, args.decoder_sample_var_floor)
        theta = np.maximum(translated_var, args.decoder_sample_var_floor)
        temperature = max(float(args.decoder_sample_temperature), args.decoder_sample_var_floor)
        theta = theta / (temperature ** 2)
        p = theta / (theta + mu)
        sampled = rng.negative_binomial(theta, p).astype(np.float32)
    else:
        raise ValueError(f"Unsupported decoder sample distribution: {args.decoder_sample_distribution}")

    if args.decoder_sample_clip_min is not None:
        sampled = np.maximum(sampled, args.decoder_sample_clip_min)
    return sampled.astype(np.float32, copy=False)


def score_translated_human_from_matrix(
    mouse_subset,
    translated,
    human_center,
    pls,
    signature_calibration,
    human_genes,
):
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
    return score_translated_human_from_matrix(
        mouse_subset,
        translated,
        human_center,
        pls,
        signature_calibration,
        human_genes,
    )


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


def score_mouse_ml_subset(
    mouse_subset,
    input_matrix,
    models,
    args,
    device,
    input_space,
    gene_space,
):
    out = ml_prediction_long_frame(
        mouse_subset.index.to_numpy(),
        models,
        input_matrix,
        args,
        device,
        input_space=input_space,
        gene_space=gene_space,
        observed=None,
    )
    return attach_mouse_metadata(out, mouse_subset)


def write_outputs(args, out_dir):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")

    out_dir.mkdir(parents=True, exist_ok=True)
    run_index = active_model_index(args)
    run_suffix = output_suffix(args)

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

    enc_h, enc_m, flow_m2h, dec_h = load_flowtransop_m2h(
        args, human_X.shape[1], mouse_X.shape[1], device
    )
    Z_human = encode_human_latent(X_human, enc_h, device, args.batch_size)
    human_latent_center = Z_human.mean(axis=0, dtype=np.float64).astype(np.float32)
    Z_human_centered = Z_human - human_latent_center

    latent_pls, used_latent_components = fit_plsr_scorer(
        Z_human_centered, y_human, n_components=args.pls_components
    )

    expression_ml_models = fit_ml_model_set(
        X_human_centered,
        y_human,
        args,
        device,
        plsr_model=pls,
        plsr_components=used_components,
    )
    orthologue_ml_models = fit_ml_model_set(
        X_human_orth_centered,
        y_human,
        args,
        device,
        plsr_model=pls_orth,
        plsr_components=used_orth_components,
    )
    latent_ml_models = fit_ml_model_set(
        Z_human_centered,
        y_human,
        args,
        device,
        plsr_model=latent_pls,
        plsr_components=used_latent_components,
    )

    loocv_paths = write_loocv_if_needed(
        args,
        out_dir,
        X_human,
        X_human_orth,
        y_human,
        human_subset.index.to_numpy(),
        Z_human=Z_human,
        device=device,
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

    human_ml_out = pd.concat(
        [
            ml_prediction_long_frame(
                human_subset.index.to_numpy(),
                expression_ml_models,
                X_human_centered,
                args,
                device,
                input_space="human_expression",
                gene_space="all_human_genes",
                observed=y_human,
            ),
            ml_prediction_long_frame(
                human_subset.index.to_numpy(),
                orthologue_ml_models,
                X_human_orth_centered,
                args,
                device,
                input_space="human_orthologue_expression",
                gene_space="orthologues",
                observed=y_human,
            ),
            ml_prediction_long_frame(
                human_subset.index.to_numpy(),
                latent_ml_models,
                Z_human_centered,
                args,
                device,
                input_space="human_latent",
                gene_space="latent",
                observed=y_human,
            ),
        ],
        ignore_index=True,
    )
    human_ml_out.insert(1, "fold", run_index)
    human_ml_out.insert(2, "model_source", args.model_source)
    if args.model_source == "full_ensemble":
        human_ml_out.insert(3, "ensemble_id", args.ensemble_id)
    human_ml_out = human_ml_out.merge(
        human_subset[["characteristics_ch1", "source_name_ch1", "title"]],
        left_on="sample_id",
        right_index=True,
        how="left",
    )

    translated_outputs = {}
    translated_orth_outputs = {}
    raw_orth_outputs = {}
    mouse_ml_outputs = {}
    decoder_sample_outputs = {}
    for cohort_name, subset in mouse_subsets.items():
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
        mouse_rows = rows_for_ids(subset.index.tolist(), mouse_ids, "mouse")
        translated, translated_var, translated_latent = translate_mouse_to_human_distribution_and_latent(
            mouse_X,
            mouse_rows,
            enc_m,
            flow_m2h,
            dec_h,
            device,
            args.batch_size,
        )
        translated_outputs[cohort_name] = score_translated_human_from_matrix(
            subset,
            translated,
            human_center,
            pls,
            signature_calibration,
            human_genes,
        )
        translated_orth_outputs[cohort_name] = score_translated_orthologue_from_matrix(
            subset,
            translated,
            human_orth_idx,
            human_orth_genes,
            human_orth_center,
            pls_orth,
            orth_signature_calibration,
        )
        translated_centered = translated - human_center
        translated_orth_centered = translated[:, human_orth_idx] - human_orth_center
        translated_latent_centered = translated_latent - human_latent_center
        X_mouse_orth = load_rows(mouse_X, mouse_rows)[:, mouse_orth_idx]
        X_mouse_orth_centered = X_mouse_orth - human_orth_center

        cohort_ml = pd.concat(
            [
                score_mouse_ml_subset(
                    subset,
                    translated_centered,
                    expression_ml_models,
                    args,
                    device,
                    input_space="translated_human",
                    gene_space="all_human_genes",
                ),
                score_mouse_ml_subset(
                    subset,
                    translated_orth_centered,
                    orthologue_ml_models,
                    args,
                    device,
                    input_space="translated_orthologues",
                    gene_space="orthologues",
                ),
                score_mouse_ml_subset(
                    subset,
                    X_mouse_orth_centered,
                    orthologue_ml_models,
                    args,
                    device,
                    input_space="raw_mouse_orthologues",
                    gene_space="orthologues",
                ),
                score_mouse_ml_subset(
                    subset,
                    translated_latent_centered,
                    latent_ml_models,
                    args,
                    device,
                    input_space="translated_human_latent",
                    gene_space="latent",
                ),
            ],
            ignore_index=True,
        )
        cohort_ml.insert(1, "fold", run_index)
        cohort_ml.insert(2, "model_source", args.model_source)
        if args.model_source == "full_ensemble":
            cohort_ml.insert(3, "ensemble_id", args.ensemble_id)
        mouse_ml_outputs[cohort_name] = cohort_ml

        decoder_draws, decoder_summary = score_decoder_sampled_translations(
            subset,
            translated,
            translated_var,
            human_center,
            human_genes,
            human_orth_idx,
            human_orth_genes,
            human_orth_center,
            expression_ml_models,
            orthologue_ml_models,
            signature_calibration,
            orth_signature_calibration,
            args,
            device,
            cohort_name,
        )
        if not decoder_draws.empty:
            decoder_draws.insert(1, "fold", run_index)
            decoder_draws.insert(2, "model_source", args.model_source)
            decoder_summary.insert(1, "fold", run_index)
            decoder_summary.insert(2, "model_source", args.model_source)
            if args.model_source == "full_ensemble":
                decoder_draws.insert(3, "ensemble_id", args.ensemble_id)
                decoder_summary.insert(3, "ensemble_id", args.ensemble_id)
        decoder_sample_outputs[cohort_name] = (decoder_draws, decoder_summary)

    summary = pd.DataFrame(
        [
            {
                "fold": run_index,
                "model_source": args.model_source,
                "ensemble_id": int(args.ensemble_id) if args.model_source == "full_ensemble" else np.nan,
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
                "latent_features": int(Z_human_centered.shape[1]),
                "latent_pls_components_used": int(used_latent_components),
                "decoder_sample_n": int(args.decoder_sample_n),
                "decoder_sample_distribution": args.decoder_sample_distribution,
                "decoder_sample_temperature": float(args.decoder_sample_temperature),
                "decoder_sample_var_floor": float(args.decoder_sample_var_floor),
                "decoder_sample_clip_min": args.decoder_sample_clip_min,
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
    ml_summary = pd.concat(
        [
            summarize_ml_model_set(
                expression_ml_models,
                X_human_centered,
                y_human,
                args,
                device,
                "human_expression",
                "all_human_genes",
            ),
            summarize_ml_model_set(
                orthologue_ml_models,
                X_human_orth_centered,
                y_human,
                args,
                device,
                "human_orthologue_expression",
                "orthologues",
            ),
            summarize_ml_model_set(
                latent_ml_models,
                Z_human_centered,
                y_human,
                args,
                device,
                "human_latent",
                "latent",
            ),
        ],
        ignore_index=True,
    )
    ml_summary.insert(0, "fold", run_index)
    ml_summary.insert(1, "model_source", args.model_source)
    if args.model_source == "full_ensemble":
        ml_summary.insert(2, "ensemble_id", args.ensemble_id)

    suffix = run_suffix
    human_path = out_dir / f"human_govaere_plsr_scores_{suffix}.csv"
    human_orth_path = out_dir / f"human_govaere_orthologue_plsr_scores_{suffix}.csv"
    summary_path = out_dir / f"plsr_summary_{suffix}.csv"
    human_signature_path = out_dir / f"human_govaere_signature_scores_{suffix}.csv"
    human_orth_signature_path = out_dir / f"human_govaere_orthologue_signature_scores_{suffix}.csv"
    signature_summary_path = out_dir / f"signature_summary_{suffix}.csv"
    human_ml_path = out_dir / f"human_govaere_ml_model_scores_{suffix}.csv"
    ml_summary_path = out_dir / f"ml_model_summary_{suffix}.csv"
    signature_regulon_path = out_dir / "signature_regulon.csv"
    orthologue_map_path = out_dir / "orthologue_gene_map.csv"

    human_out.to_csv(human_path, index=False)
    human_orth_out.to_csv(human_orth_path, index=False)
    summary.to_csv(summary_path, index=False)
    human_sig_out.to_csv(human_signature_path, index=False)
    human_orth_sig_out.to_csv(human_orth_signature_path, index=False)
    signature_summary.to_csv(signature_summary_path, index=False)
    human_ml_out.to_csv(human_ml_path, index=False)
    ml_summary.to_csv(ml_summary_path, index=False)
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
        human_ml_path,
        ml_summary_path,
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

    for cohort_name, (plsr_out, sig_out) in translated_orth_outputs.items():
        plsr_path = out_dir / f"{cohort_name}_translated_orthologue_plsr_scores_{suffix}.csv"
        sig_path = out_dir / f"{cohort_name}_translated_orthologue_signature_scores_{suffix}.csv"
        plsr_out.to_csv(plsr_path, index=False)
        sig_out.to_csv(sig_path, index=False)
        output_paths.extend([plsr_path, sig_path])

    for cohort_name, (plsr_out, sig_out) in raw_orth_outputs.items():
        plsr_path = out_dir / f"{cohort_name}_raw_orthologue_plsr_scores_{suffix}.csv"
        sig_path = out_dir / f"{cohort_name}_raw_orthologue_signature_scores_{suffix}.csv"
        plsr_out.to_csv(plsr_path, index=False)
        sig_out.to_csv(sig_path, index=False)
        output_paths.extend([plsr_path, sig_path])

    for cohort_name, ml_out in mouse_ml_outputs.items():
        ml_path = out_dir / f"{cohort_name}_ml_model_scores_{suffix}.csv"
        ml_out.to_csv(ml_path, index=False)
        output_paths.append(ml_path)

    for cohort_name, (decoder_draws, decoder_summary) in decoder_sample_outputs.items():
        if decoder_summary.empty:
            continue
        summary_path = out_dir / f"{cohort_name}_decoder_sampled_score_summary_{suffix}.csv"
        decoder_summary.to_csv(summary_path, index=False)
        output_paths.append(summary_path)
        if args.save_decoder_sample_draws:
            draws_path = out_dir / f"{cohort_name}_decoder_sampled_score_draws_{suffix}.csv"
            decoder_draws.to_csv(draws_path, index=False)
            output_paths.append(draws_path)

    for path in output_paths:
        print(f"Wrote: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_source",
        choices=["fold", "full_ensemble"],
        default="fold",
        help="Use CV fold checkpoints or full-data ensemble checkpoints.",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--ensemble_id",
        type=int,
        default=0,
        help="Full-data ensemble member to load when --model_source full_ensemble.",
    )
    parser.add_argument(
        "--full_model_prefix",
        default="full_ensemble",
        help="Checkpoint prefix for full ensemble models, before _<ensemble_id>_normal.pt.",
    )
    parser.add_argument(
        "--output_suffix",
        default=None,
        help="Override output filename suffix. Defaults to foldN or ensembleN.",
    )
    parser.add_argument(
        "--ensemble_suffix_as_fold",
        action="store_true",
        help="In full_ensemble mode, write files as fold<ensemble_id> for compatibility with the plotting script.",
    )
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
        "--decoder_sample_n",
        type=int,
        default=25,
        help="Number of decoder-expression draws per mouse sample. Use 0 to disable sampled scoring.",
    )
    parser.add_argument(
        "--decoder_sample_distribution",
        choices=["gaussian", "negative_binomial"],
        default="gaussian",
        help="Distribution used for decoder sampling. ARCHS4 fold/full models were trained with gaussian decoders.",
    )
    parser.add_argument(
        "--decoder_sample_temperature",
        type=float,
        default=1.0,
        help="Multiplier for sampled decoder spread. For gaussian this scales sd; for NB this scales overdispersion.",
    )
    parser.add_argument(
        "--decoder_sample_var_floor",
        type=float,
        default=1e-6,
        help="Minimum decoder variance/dispersion used during sampled scoring.",
    )
    parser.add_argument(
        "--decoder_sample_clip_min",
        type=float,
        default=0.0,
        help="Minimum expression value after sampling. Defaults to 0 because decoder means are non-negative.",
    )
    parser.add_argument(
        "--no_decoder_sample_clip",
        action="store_const",
        const=None,
        dest="decoder_sample_clip_min",
        help="Do not clip sampled decoder expression values.",
    )
    parser.add_argument(
        "--decoder_sample_seed",
        type=int,
        default=1729,
        help="Base random seed for decoder sampled scoring.",
    )
    parser.add_argument(
        "--save_decoder_sample_draws",
        action="store_true",
        default=True,
        help="Save per-draw decoder sampled scores in addition to per-sample summaries.",
    )
    parser.add_argument(
        "--skip_decoder_sample_draws",
        action="store_false",
        dest="save_decoder_sample_draws",
        help="Only save decoder sampled score summaries, not every draw.",
    )
    parser.add_argument("--skip_svm_rbf", action="store_true", help="Skip RBF-SVM scorers.")
    parser.add_argument("--svm_cv", type=int, default=5)
    parser.add_argument("--svm_jobs", type=int, default=-1)
    parser.add_argument(
        "--skip_loocv",
        action="store_true",
        help="Skip all human leave-one-out predictions.",
    )
    parser.add_argument(
        "--skip_ml_loocv",
        action="store_true",
        help="Skip RBF-SVM human leave-one-out predictions.",
    )
    parser.add_argument(
        "--loocv_write_fold",
        type=int,
        default=0,
        help="Only this fold/ensemble index writes the non-run-specific LOOCV CSVs.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.splits_dir = args.splits_dir or args.data_dir / "splits"
    args.preproc_dir = args.preproc_dir or args.data_dir / "preprocessed"
    args.model_dir = args.model_dir or args.data_dir / "models"
    args.out_dir = args.out_dir or args.data_dir / "evaluation" / "liver_mas_fibrosis"
    if args.decoder_sample_n < 0:
        raise ValueError("--decoder_sample_n must be >= 0.")
    if args.decoder_sample_temperature < 0:
        raise ValueError("--decoder_sample_temperature must be >= 0.")
    if args.decoder_sample_var_floor <= 0:
        raise ValueError("--decoder_sample_var_floor must be > 0.")
    return args


def main():
    args = parse_args()
    write_outputs(args, args.out_dir)


if __name__ == "__main__":
    main()
