#!/usr/bin/env python3
"""
Final liver MAS/fibrosis scoring with expression-level ensemble averaging.

This script intentionally keeps only two mouse studies:
  * GSE140742 Nlrp3A350V / GS-444217
  * GSE269493 CDAA-HFD / Lanifibranor

It averages model-predicted gene expression across full-ensemble ARCHS4 models
before scoring. Two PLSR model families are trained:
  1. raw_human_plsr: observed Govaere human expression.
  2. reconstructed_human_plsr: Govaere human expression reconstructed through
     each human autoencoder, averaged across ensemble members.

For each mouse dataset, the script scores:
  * expression-mean translated all human genes,
  * expression-mean translated human orthologues,
  * raw mouse orthologues, without translation or correction.

The raw mouse orthologue scores are computed once per dataset for each PLSR
family and are also written to one raw-orthologue-only CSV per dataset.
"""

import argparse
import re
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

import evaluate_translation as et
from models import ElementWiseLinear, Flow, SimpleEncoder, VarDecoder


DATA_DIR = Path("../archs4")
OUT_DIR = DATA_DIR / "evaluation" / "liver_mas_fibrosis_final_expression_mean"
PLS_COMPONENTS = 8
HUMAN_GSE = "GSE135251"
MOUSE_GS_GSE = "GSE140742"
MOUSE_CDAA_GSE = "GSE269493"

HUMAN_META_COLS = ["characteristics_ch1", "source_name_ch1", "title"]
MOUSE_COHORTS = ("mouse_nlrp3", "mouse_cdaa_lanifibranor")


def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return path


def parse_int_ranges(text):
    values = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, stop = chunk.split("-", 1)
            values.extend(range(int(start), int(stop) + 1))
        else:
            values.append(int(chunk))
    if not values:
        raise ValueError("No integer IDs were parsed.")
    return sorted(dict.fromkeys(values))


def read_metadata(path):
    metadata = pd.read_csv(require_file(path), index_col=0, low_memory=False)
    metadata.index = metadata.index.astype(str)
    return metadata


def rows_for_ids(sample_ids, all_ids, label):
    row_lookup = {str(sample_id): i for i, sample_id in enumerate(np.asarray(all_ids).astype(str))}
    missing = [str(sample_id) for sample_id in sample_ids if str(sample_id) not in row_lookup]
    if missing:
        raise KeyError(f"{len(missing)} {label} sample IDs were not found; first missing: {missing[:5]}")
    return np.asarray([row_lookup[str(sample_id)] for sample_id in sample_ids], dtype=np.int64)


def load_rows(matrix, rows):
    return np.asarray(matrix[np.asarray(rows, dtype=np.int64), :], dtype=np.float32)


def _sample_frame(metadata, sample_ids):
    ids = pd.Index(np.asarray(sample_ids).astype(str))
    keep = metadata.index.intersection(ids)
    return metadata.loc[keep].copy()


def _text_column(frame):
    cols = [c for c in ["characteristics_ch1", "source_name_ch1", "title"] if c in frame.columns]
    if not cols:
        return pd.Series("", index=frame.index)
    return frame[cols].fillna("").astype(str).agg(" ".join, axis=1)


def _extract_characteristic(text, field):
    pattern = rf"{re.escape(field)}\s*:\s*([^,]+)"
    return text.astype(str).str.extract(pattern, flags=re.IGNORECASE, expand=False).str.strip()


def select_human_govaere(metadata, sample_ids, human_gse=HUMAN_GSE):
    frame = _sample_frame(metadata, sample_ids)
    characteristics = frame.get("characteristics_ch1", pd.Series("", index=frame.index)).fillna("").astype(str)
    mask = characteristics.str.contains(r"nas score\s*:", case=False, regex=True, na=False)
    mask &= characteristics.str.contains(r"fibrosis stage\s*:", case=False, regex=True, na=False)
    out = frame.loc[mask].copy()
    out["nas_score"] = pd.to_numeric(
        _extract_characteristic(out["characteristics_ch1"].fillna(""), "nas score"),
        errors="coerce",
    )
    out["fibrosis_stage"] = pd.to_numeric(
        _extract_characteristic(out["characteristics_ch1"].fillna(""), "fibrosis stage"),
        errors="coerce",
    )
    out = out.dropna(subset=["nas_score", "fibrosis_stage"])
    if out.empty:
        raise ValueError(f"No Govaere human samples with NAS/fibrosis scores were selected for {human_gse}.")
    return out


def select_mouse_nlrp3(metadata, sample_ids, mouse_gse=MOUSE_GS_GSE):
    frame = _sample_frame(metadata, sample_ids)
    characteristics = frame.get("characteristics_ch1", pd.Series("", index=frame.index)).fillna("").astype(str)
    lower = characteristics.str.lower()
    liver = lower.str.contains("tissue: liver", regex=False, na=False)
    wt = lower.str.contains("genotype/variation: wt", regex=False, na=False)
    nlrp3 = lower.str.contains("nlrp3a350v", regex=False, na=False)
    placebo = lower.str.contains("control chow+placebo", regex=False, na=False)
    gs = lower.str.contains("gs-444217", regex=False, na=False)

    healthy = liver & wt & placebo
    control = liver & nlrp3 & placebo
    treated = liver & nlrp3 & gs
    out = frame.loc[healthy | control | treated].copy()
    out["mouse_model"] = np.where(nlrp3.loc[out.index], "Nlrp3A350V", "WT")
    out["mouse_genotype"] = np.where(nlrp3.loc[out.index], "Nlrp3A350V/+CreT", "WT")
    out["mouse_treatment"] = np.select(
        [healthy.loc[out.index], control.loc[out.index], treated.loc[out.index]],
        ["WT Control chow+placebo", "Nlrp3 Control chow+placebo", "GS-444217"],
        default="other",
    )
    out["time_point"] = "not_reported"
    if out.empty:
        raise ValueError(f"No Nlrp3A350V / GS-444217 mouse samples were selected for {mouse_gse}.")
    return out


def select_mouse_cdaa_lanifibranor(metadata, sample_ids, mouse_gse=MOUSE_CDAA_GSE):
    frame = _sample_frame(metadata, sample_ids)
    characteristics = frame.get("characteristics_ch1", pd.Series("", index=frame.index)).fillna("").astype(str)
    titles = frame.get("title", pd.Series("", index=frame.index)).fillna("").astype(str)
    lower = characteristics.str.lower()
    title_lower = titles.str.lower()
    liver = lower.str.contains("tissue: liver", regex=False, na=False)
    chow = (
        liver
        & lower.str.contains("treatment: chow", regex=False, na=False)
        & title_lower.str.contains("liver, chow, rep", regex=False, na=False)
    )
    vehicle = liver & lower.str.contains("treatment: cdaa-hfd vehicle", regex=False, na=False)
    lanifibranor = liver & lower.str.contains("treatment: cdaa-hfd lanifibranor", regex=False, na=False)

    out = frame.loc[chow | vehicle | lanifibranor].copy()
    out["mouse_model"] = np.select(
        [chow.loc[out.index], vehicle.loc[out.index] | lanifibranor.loc[out.index]],
        ["Chow", "CDAA-HFD"],
        default="other",
    )
    out["mouse_genotype"] = "C57BL/6J"
    out["mouse_treatment"] = np.select(
        [chow.loc[out.index], vehicle.loc[out.index], lanifibranor.loc[out.index]],
        ["Chow", "CDAA-HFD vehicle", "Lanifibranor"],
        default="other",
    )
    out["time_point"] = np.where(chow.loc[out.index], "healthy_reference", "study_endpoint")
    if out.empty:
        raise ValueError(f"No CDAA-HFD / Lanifibranor mouse samples were selected for {mouse_gse}.")
    return out


def _activation(name):
    return {
        "LeakyReLU": torch.nn.LeakyReLU(0.01),
        "ReLU": torch.nn.ReLU(),
        "ELU": torch.nn.ELU(),
        "Sigmoid": torch.nn.Sigmoid(),
    }[name]


def _model_params(saved_args):
    params = dict(saved_args)
    params["encoder_activation"] = _activation(params["encoder_activation"])
    params["decoder_activation"] = _activation(params["decoder_activation"])
    return params


def _build_models(saved_args, human_dim, mouse_dim, device):
    params = _model_params(saved_args)
    encoder_human = torch.nn.Sequential(
        ElementWiseLinear(human_dim),
        SimpleEncoder(
            human_dim,
            params["encoder_1_hiddens"],
            params["latent_dim"],
            dropRate=params["dropout_encoder"],
            bn=params["bn_encoder"],
            activation=params["encoder_activation"],
            dropIn=params["dropout_input_encoder"],
            dtype=torch.float,
        ),
    ).to(device)
    encoder_mouse = torch.nn.Sequential(
        ElementWiseLinear(mouse_dim),
        SimpleEncoder(
            mouse_dim,
            params["encoder_2_hiddens"],
            params["latent_dim"],
            dropRate=params["dropout_encoder"],
            bn=params["bn_encoder"],
            activation=params["encoder_activation"],
            dropIn=params["dropout_input_encoder"],
            dtype=torch.float,
        ),
    ).to(device)
    decoder_human = VarDecoder(
        params["latent_dim"],
        params["decoder_1_hiddens"],
        human_dim,
        dropRate=params["dropout_decoder"],
        bn=params["bn_decoder"],
        activation=params["decoder_activation"],
        dropIn=params["dropout_input_decoder"],
        loss="gauss",
        dtype=torch.float,
    ).to(device)
    decoder_mouse = VarDecoder(
        params["latent_dim"],
        params["decoder_2_hiddens"],
        mouse_dim,
        dropRate=params["dropout_decoder"],
        bn=params["bn_decoder"],
        activation=params["decoder_activation"],
        dropIn=params["dropout_input_decoder"],
        loss="gauss",
        dtype=torch.float,
    ).to(device)
    flow_m2h = Flow(params["latent_dim"], params["latent_dim"] // 2, dtype=torch.float).to(device)
    return encoder_human, encoder_mouse, decoder_human, decoder_mouse, flow_m2h


def load_flowtransop_m2h(args, human_dim, mouse_dim, device):
    if args.model_source != "full_ensemble":
        raise ValueError("score_liver_mas_fibrosis_final_expression_mean.py expects --model_source full_ensemble.")
    normal_path = require_file(
        args.model_dir / f"{args.full_model_prefix}_{int(args.ensemble_id)}_normal.pt"
    )
    ckpt = torch.load(normal_path, map_location=device, weights_only=False)
    enc_h, enc_m, dec_h, dec_m, flow_m2h = _build_models(ckpt["args"], human_dim, mouse_dim, device)
    enc_h.load_state_dict(ckpt["encoder_human"])
    enc_m.load_state_dict(ckpt["encoder_mouse"])
    dec_h.load_state_dict(ckpt["decoder_human"])
    dec_m.load_state_dict(ckpt["decoder_mouse"])

    m2h_path = args.model_dir / f"{args.full_model_prefix}_{int(args.ensemble_id)}_normal_m2h.pt"
    if m2h_path.exists():
        m2h_ckpt = torch.load(m2h_path, map_location=device, weights_only=False)
        flow_m2h.load_state_dict(m2h_ckpt["flow_m2h"])
    else:
        flow_m2h.load_state_dict(ckpt["flow_m2h"])

    for model in (enc_h, enc_m, dec_h, dec_m, flow_m2h):
        model.eval()
    return enc_h, enc_m, flow_m2h, dec_h


@torch.no_grad()
def translate_mouse_to_human_distribution_and_latent(mouse_matrix, rows, enc_m, flow_m2h, dec_h, device, batch_size):
    out_dim = int(dec_h.out_mu.out_features)
    translated_mu = np.empty((len(rows), out_dim), dtype=np.float32)
    translated_sd = np.empty((len(rows), out_dim), dtype=np.float32)
    latent = []
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        block = np.asarray(mouse_matrix[rows[start:stop], :], dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(block)).to(device)
        z = flow_m2h(enc_m(x))
        mu, logvar = dec_h(z)
        translated_mu[start:stop] = mu.cpu().numpy()
        translated_sd[start:stop] = torch.exp(0.5 * logvar).cpu().numpy()
        latent.append(z.cpu().numpy())
    return translated_mu, translated_sd, np.vstack(latent).astype(np.float32)


def fit_plsr_scorer(X, y, requested_components):
    components = max(1, min(int(requested_components), X.shape[0] - 1, X.shape[1]))
    model = PLSRegression(n_components=components, scale=False)
    model.fit(X, y)
    return model, components


def loocv_plsr_predictions(X, y, sample_ids, requested_components, feature_space):
    rows = []
    for i, sample_id in enumerate(np.asarray(sample_ids).astype(str)):
        train_mask = np.ones(X.shape[0], dtype=bool)
        train_mask[i] = False
        center = X[train_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
        model, components = fit_plsr_scorer(X[train_mask] - center, y[train_mask], requested_components)
        pred = model.predict((X[i : i + 1] - center))[0]
        rows.append(
            {
                "sample_id": sample_id,
                "feature_space": feature_space,
                "pls_components_used": int(components),
                "observed_nas_score": float(y[i, 0]),
                "observed_fibrosis_stage": float(y[i, 1]),
                "predicted_nas_score": float(pred[0]),
                "predicted_fibrosis_stage": float(pred[1]),
            }
        )
    return pd.DataFrame(rows)


def attach_mouse_metadata(frame, mouse_subset):
    cols = [
        "dataset",
        "mouse_model",
        "mouse_genotype",
        "mouse_treatment",
        "time_point",
        "characteristics_ch1",
        "source_name_ch1",
        "title",
    ]
    meta = mouse_subset[[c for c in cols if c in mouse_subset.columns]].copy()
    return frame.merge(meta, left_on="sample_id", right_index=True, how="left")


base = types.SimpleNamespace(
    PLS_COMPONENTS=PLS_COMPONENTS,
    HUMAN_GSE=HUMAN_GSE,
    MOUSE_GS_GSE=MOUSE_GS_GSE,
    MOUSE_CDAA_GSE=MOUSE_CDAA_GSE,
    require_file=require_file,
    parse_int_ranges=parse_int_ranges,
    read_metadata=read_metadata,
    select_human_govaere=select_human_govaere,
    select_mouse_nlrp3=select_mouse_nlrp3,
    select_mouse_cdaa_lanifibranor=select_mouse_cdaa_lanifibranor,
    rows_for_ids=rows_for_ids,
    load_rows=load_rows,
    load_flowtransop_m2h=load_flowtransop_m2h,
    translate_mouse_to_human_distribution_and_latent=translate_mouse_to_human_distribution_and_latent,
    fit_plsr_scorer=fit_plsr_scorer,
    loocv_plsr_predictions=loocv_plsr_predictions,
    attach_mouse_metadata=attach_mouse_metadata,
)


def args_for_ensemble(args, ensemble_id):
    return argparse.Namespace(
        model_source="full_ensemble",
        ensemble_id=int(ensemble_id),
        fold=0,
        full_model_prefix=args.full_model_prefix,
        model_dir=args.model_dir,
    )


@torch.no_grad()
def reconstruct_human_mean(human_matrix, human_rows, enc_h, dec_h, device, batch_size):
    out_dim = int(dec_h.out_mu.out_features)
    reconstructed = np.empty((len(human_rows), out_dim), dtype=np.float32)
    for start in range(0, len(human_rows), batch_size):
        stop = min(start + batch_size, len(human_rows))
        block = np.asarray(human_matrix[human_rows[start:stop], :], dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(block)).to(device)
        mu, _ = dec_h(enc_h(x))
        reconstructed[start:stop] = mu.cpu().numpy()
    return reconstructed


def load_context(args):
    human_X = np.load(base.require_file(args.preproc_dir / "human_test_X.npy"), mmap_mode="r")
    mouse_X = np.load(base.require_file(args.preproc_dir / "mouse_test_X.npy"), mmap_mode="r")
    human_genes = np.load(base.require_file(args.preproc_dir / "human_genes.npy"), allow_pickle=True)
    mouse_genes = np.load(base.require_file(args.preproc_dir / "mouse_genes.npy"), allow_pickle=True)
    human_ids = np.load(base.require_file(args.preproc_dir / "human_test_sample_ids.npy"), allow_pickle=True)
    mouse_ids = np.load(base.require_file(args.preproc_dir / "mouse_test_sample_ids.npy"), allow_pickle=True)

    human_meta = base.read_metadata(args.splits_dir / "liver_metadata_human.csv")
    mouse_meta = base.read_metadata(args.splits_dir / "liver_metadata_mouse.csv")

    human_subset = base.select_human_govaere(human_meta, human_ids, args.human_gse)
    mouse_subsets = {
        "mouse_nlrp3": base.select_mouse_nlrp3(mouse_meta, mouse_ids, args.mouse_gs_gse),
        "mouse_cdaa_lanifibranor": base.select_mouse_cdaa_lanifibranor(
            mouse_meta, mouse_ids, args.mouse_cdaa_gse
        ),
    }

    human_rows = base.rows_for_ids(human_subset.index.tolist(), human_ids, "human")
    y_human = human_subset[["nas_score", "fibrosis_stage"]].to_numpy(dtype=np.float32)
    human_orth_idx, mouse_orth_idx = et.build_orthologue_map(human_genes, mouse_genes)

    return {
        "human_X": human_X,
        "mouse_X": mouse_X,
        "human_genes": human_genes,
        "mouse_genes": mouse_genes,
        "human_ids": human_ids,
        "mouse_ids": mouse_ids,
        "human_subset": human_subset,
        "mouse_subsets": mouse_subsets,
        "human_rows": human_rows,
        "y_human": y_human,
        "human_orth_idx": human_orth_idx,
        "mouse_orth_idx": mouse_orth_idx,
    }


def average_ensemble_expression(args, context, device):
    ensemble_ids = base.parse_int_ranges(args.ensemble_ids)
    human_sum = None
    mouse_acc = {}

    for cohort_name, subset in context["mouse_subsets"].items():
        mouse_acc[cohort_name] = {
            "subset": subset,
            "rows": base.rows_for_ids(subset.index.tolist(), context["mouse_ids"], "mouse"),
            "translated_sum": None,
        }

    for ensemble_id in ensemble_ids:
        member_args = args_for_ensemble(args, ensemble_id)
        enc_h, enc_m, flow_m2h, dec_h = base.load_flowtransop_m2h(
            member_args,
            context["human_X"].shape[1],
            context["mouse_X"].shape[1],
            device,
        )

        reconstructed_human = reconstruct_human_mean(
            context["human_X"],
            context["human_rows"],
            enc_h,
            dec_h,
            device,
            args.batch_size,
        )
        if human_sum is None:
            human_sum = np.zeros_like(reconstructed_human, dtype=np.float64)
        human_sum += reconstructed_human.astype(np.float64)

        for acc in mouse_acc.values():
            translated_mu, _, _ = base.translate_mouse_to_human_distribution_and_latent(
                context["mouse_X"],
                acc["rows"],
                enc_m,
                flow_m2h,
                dec_h,
                device,
                args.batch_size,
            )
            if acc["translated_sum"] is None:
                acc["translated_sum"] = np.zeros_like(translated_mu, dtype=np.float64)
            acc["translated_sum"] += translated_mu.astype(np.float64)

        del enc_h, enc_m, flow_m2h, dec_h
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_models = float(len(ensemble_ids))
    averaged_human_reconstruction = (human_sum / n_models).astype(np.float32)
    averaged_mouse_translations = {
        cohort_name: (acc["translated_sum"] / n_models).astype(np.float32)
        for cohort_name, acc in mouse_acc.items()
    }

    return ensemble_ids, averaged_human_reconstruction, averaged_mouse_translations


def fit_plsr_family(X_human, y_human, human_orth_idx, args):
    center = X_human.mean(axis=0, dtype=np.float64).astype(np.float32)
    X_centered = X_human - center
    pls, components = base.fit_plsr_scorer(X_centered, y_human, args.pls_components)
    pred = pls.predict(X_centered)

    X_orth = X_human[:, human_orth_idx]
    orth_center = X_orth.mean(axis=0, dtype=np.float64).astype(np.float32)
    X_orth_centered = X_orth - orth_center
    pls_orth, orth_components = base.fit_plsr_scorer(
        X_orth_centered, y_human, args.pls_components
    )
    orth_pred = pls_orth.predict(X_orth_centered)

    return {
        "center": center,
        "X_centered": X_centered,
        "pls": pls,
        "components": int(components),
        "pred": pred,
        "orth_center": orth_center,
        "X_orth_centered": X_orth_centered,
        "pls_orth": pls_orth,
        "orth_components": int(orth_components),
        "orth_pred": orth_pred,
    }


def add_human_metadata(frame, human_subset):
    meta_cols = [col for col in HUMAN_META_COLS if col in human_subset.columns]
    return frame.merge(
        human_subset[meta_cols],
        left_on="sample_id",
        right_index=True,
        how="left",
    )


def human_prediction_frame(sample_ids, y, pred, model_family, scoring_space, feature_space):
    return pd.DataFrame(
        {
            "sample_id": np.asarray(sample_ids).astype(str),
            "model_family": model_family,
            "scoring_space": scoring_space,
            "feature_space": feature_space,
            "observed_nas_score": y[:, 0],
            "observed_fibrosis_stage": y[:, 1],
            "predicted_nas_score": pred[:, 0],
            "predicted_fibrosis_stage": pred[:, 1],
        }
    )


def mouse_prediction_frame(sample_ids, pred, model_family, scoring_space, feature_space):
    return pd.DataFrame(
        {
            "sample_id": np.asarray(sample_ids).astype(str),
            "model_family": model_family,
            "scoring_space": scoring_space,
            "feature_space": feature_space,
            "predicted_nas_score": pred[:, 0],
            "predicted_fibrosis_stage": pred[:, 1],
        }
    )


def attach_mouse(frame, mouse_subset):
    return base.attach_mouse_metadata(frame, mouse_subset)


def score_mouse_dataset(context, cohort_name, translated_mean, raw_family, recon_family):
    subset = context["mouse_subsets"][cohort_name]
    mouse_rows = base.rows_for_ids(subset.index.tolist(), context["mouse_ids"], "mouse")
    X_mouse_orth = base.load_rows(context["mouse_X"], mouse_rows)[:, context["mouse_orth_idx"]]
    sample_ids = subset.index.to_numpy()

    rows = []
    raw_specs = [
        (
            "raw_human_plsr",
            "translated_all_genes",
            "all_human_genes",
            raw_family["pls"].predict(translated_mean - raw_family["center"]),
        ),
        (
            "raw_human_plsr",
            "translated_orthologues",
            "orthologues",
            raw_family["pls_orth"].predict(
                translated_mean[:, context["human_orth_idx"]] - raw_family["orth_center"]
            ),
        ),
        (
            "raw_human_plsr",
            "raw_mouse_orthologues",
            "orthologues",
            raw_family["pls_orth"].predict(X_mouse_orth - raw_family["orth_center"]),
        ),
    ]
    recon_specs = [
        (
            "reconstructed_human_plsr",
            "translated_all_genes",
            "all_human_genes",
            recon_family["pls"].predict(translated_mean - recon_family["center"]),
        ),
        (
            "reconstructed_human_plsr",
            "translated_orthologues",
            "orthologues",
            recon_family["pls_orth"].predict(
                translated_mean[:, context["human_orth_idx"]] - recon_family["orth_center"]
            ),
        ),
        (
            "reconstructed_human_plsr",
            "raw_mouse_orthologues",
            "orthologues",
            recon_family["pls_orth"].predict(X_mouse_orth - recon_family["orth_center"]),
        ),
    ]

    for model_family, scoring_space, feature_space, pred in raw_specs + recon_specs:
        rows.append(
            mouse_prediction_frame(sample_ids, pred, model_family, scoring_space, feature_space)
        )

    out = pd.concat(rows, ignore_index=True)
    return attach_mouse(out, subset)


def training_summary(model_family, family, y_human, n_ensemble_models):
    rows = []
    specs = [
        ("training_all_genes", "all_human_genes", family["X_centered"], family["pred"], family["components"]),
        (
            "training_orthologues",
            "orthologues",
            family["X_orth_centered"],
            family["orth_pred"],
            family["orth_components"],
        ),
    ]
    for scoring_space, feature_space, X, pred, components in specs:
        rows.append(
            {
                "model_family": model_family,
                "scoring_space": scoring_space,
                "feature_space": feature_space,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "n_ensemble_models": int(n_ensemble_models),
                "pls_components_used": int(components),
                "train_rmse_nas": float(mean_squared_error(y_human[:, 0], pred[:, 0]) ** 0.5),
                "train_rmse_fibrosis": float(mean_squared_error(y_human[:, 1], pred[:, 1]) ** 0.5),
                "train_r2_nas": float(r2_score(y_human[:, 0], pred[:, 0])),
                "train_r2_fibrosis": float(r2_score(y_human[:, 1], pred[:, 1])),
            }
        )
    return pd.DataFrame(rows)


def add_loocv_metadata(frame, model_family, scoring_space, feature_space, n_ensemble_models):
    frame = frame.copy()
    frame.insert(1, "model_family", model_family)
    frame.insert(2, "scoring_space", scoring_space)
    frame.insert(3, "feature_space", feature_space)
    frame.insert(4, "n_ensemble_models", int(n_ensemble_models))
    return frame


def write_loocv_outputs(args, out_dir, context, raw_human_X, reconstructed_human_X, n_ensemble_models):
    if args.skip_loocv:
        return []
    y = context["y_human"]
    ids = context["human_subset"].index.to_numpy()
    human_orth_idx = context["human_orth_idx"]

    specs = [
        (
            "raw_human_plsr",
            "all_human_genes",
            "all_human_genes",
            raw_human_X,
            "human_govaere_raw_plsr_loocv_all_genes.csv",
        ),
        (
            "raw_human_plsr",
            "orthologues",
            "orthologues",
            raw_human_X[:, human_orth_idx],
            "human_govaere_raw_plsr_loocv_orthologues.csv",
        ),
        (
            "reconstructed_human_plsr",
            "all_human_genes",
            "all_human_genes",
            reconstructed_human_X,
            "human_govaere_reconstructed_plsr_loocv_all_genes.csv",
        ),
        (
            "reconstructed_human_plsr",
            "orthologues",
            "orthologues",
            reconstructed_human_X[:, human_orth_idx],
            "human_govaere_reconstructed_plsr_loocv_orthologues.csv",
        ),
    ]

    paths = []
    for model_family, scoring_space, feature_space, X, filename in specs:
        loocv = base.loocv_plsr_predictions(X, y, ids, args.pls_components, feature_space)
        loocv = add_loocv_metadata(
            loocv, model_family, scoring_space, feature_space, n_ensemble_models
        )
        path = out_dir / filename
        loocv.to_csv(path, index=False)
        paths.append(path)
    return paths


def write_outputs(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    context = load_context(args)
    ensemble_ids, reconstructed_human_mean, translated_means = average_ensemble_expression(
        args, context, device
    )
    n_models = len(ensemble_ids)

    raw_human_X = base.load_rows(context["human_X"], context["human_rows"])
    y_human = context["y_human"]

    raw_family = fit_plsr_family(raw_human_X, y_human, context["human_orth_idx"], args)
    recon_family = fit_plsr_family(
        reconstructed_human_mean, y_human, context["human_orth_idx"], args
    )

    human_frames = [
        human_prediction_frame(
            context["human_subset"].index.to_numpy(),
            y_human,
            raw_family["pred"],
            "raw_human_plsr",
            "training_all_genes",
            "all_human_genes",
        ),
        human_prediction_frame(
            context["human_subset"].index.to_numpy(),
            y_human,
            raw_family["orth_pred"],
            "raw_human_plsr",
            "training_orthologues",
            "orthologues",
        ),
        human_prediction_frame(
            context["human_subset"].index.to_numpy(),
            y_human,
            recon_family["pred"],
            "reconstructed_human_plsr",
            "training_all_genes",
            "all_human_genes",
        ),
        human_prediction_frame(
            context["human_subset"].index.to_numpy(),
            y_human,
            recon_family["orth_pred"],
            "reconstructed_human_plsr",
            "training_orthologues",
            "orthologues",
        ),
    ]
    human_scores = add_human_metadata(pd.concat(human_frames, ignore_index=True), context["human_subset"])
    human_scores.insert(1, "ensemble_ids", ",".join(str(x) for x in ensemble_ids))
    human_scores.insert(2, "n_ensemble_models", n_models)

    summary = pd.concat(
        [
            training_summary("raw_human_plsr", raw_family, y_human, n_models),
            training_summary("reconstructed_human_plsr", recon_family, y_human, n_models),
        ],
        ignore_index=True,
    )
    summary.insert(0, "ensemble_ids", ",".join(str(x) for x in ensemble_ids))

    output_paths = []
    human_path = args.out_dir / "human_govaere_final_plsr_training_predictions.csv"
    summary_path = args.out_dir / "final_plsr_training_summary.csv"
    human_scores.to_csv(human_path, index=False)
    summary.to_csv(summary_path, index=False)
    output_paths.extend([human_path, summary_path])

    output_paths.extend(
        write_loocv_outputs(
            args,
            args.out_dir,
            context,
            raw_human_X,
            reconstructed_human_mean,
            n_models,
        )
    )

    for cohort_name in MOUSE_COHORTS:
        scores = score_mouse_dataset(
            context,
            cohort_name,
            translated_means[cohort_name],
            raw_family,
            recon_family,
        )
        scores.insert(1, "ensemble_ids", ",".join(str(x) for x in ensemble_ids))
        scores.insert(2, "n_ensemble_models", n_models)
        all_path = args.out_dir / f"{cohort_name}_final_expression_mean_plsr_scores.csv"
        raw_path = args.out_dir / f"{cohort_name}_raw_mouse_orthologue_plsr_scores.csv"
        scores.to_csv(all_path, index=False)
        scores.loc[scores["scoring_space"] == "raw_mouse_orthologues"].to_csv(
            raw_path, index=False
        )
        output_paths.extend([all_path, raw_path])

    for path in output_paths:
        print(f"Wrote: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_ids", default="0-9")
    parser.add_argument("--full_model_prefix", default="full_ensemble")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--splits_dir", type=Path, default=None)
    parser.add_argument("--preproc_dir", type=Path, default=None)
    parser.add_argument("--model_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--human_gse", default=base.HUMAN_GSE)
    parser.add_argument("--mouse_gs_gse", default=base.MOUSE_GS_GSE)
    parser.add_argument("--mouse_cdaa_gse", default=base.MOUSE_CDAA_GSE)
    parser.add_argument("--pls_components", type=int, default=PLS_COMPONENTS)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--skip_loocv", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.splits_dir = args.splits_dir or args.data_dir / "splits"
    args.preproc_dir = args.preproc_dir or args.data_dir / "preprocessed"
    args.model_dir = args.model_dir or args.data_dir / "models"
    args.out_dir = args.out_dir or OUT_DIR
    base.parse_int_ranges(args.ensemble_ids)
    return args


def main():
    write_outputs(parse_args())


if __name__ == "__main__":
    main()
