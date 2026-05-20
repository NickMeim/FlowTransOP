#!/usr/bin/env python3
"""
Hallmark ssGSEA analysis on ensemble-averaged translated mouse liver expression.

This workflow uses the same final translated all-gene expression space used for
the MAS/fibrosis disease scoring:
  * full-ensemble ARCHS4 mouse->human translated decoder means,
  * averaged at the gene-expression level across ensemble members,
  * restricted to the two retained mouse studies:
      - GSE140742 Nlrp3A350V / GS-444217
      - GSE269493 CDAA-HFD / Lanifibranor

For each biological sample it computes Hallmark single-sample GSEA-like rank
enrichment on the translated human gene-expression space.

The script saves per-sample ssGSEA matrices, long-form ssGSEA values,
drug-vs-disease contrasts, top-feature summaries, and PNG visualizations of
the main treatment differences.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

import score_liver_mas_fibrosis_final_expression_mean as scoring


DATA_DIR = scoring.DATA_DIR
OUT_DIR = DATA_DIR / "evaluation" / "liver_mas_fibrosis_hallmark_ssgsea"
MOUSE_COHORTS = scoring.MOUSE_COHORTS
HUMAN_GSE = "GSE135251"
MOUSE_GS_GSE = "GSE140742"
MOUSE_CDAA_GSE = "GSE269493"

COHORT_SPECS = {
    "mouse_nlrp3": {
        "dataset_label": "GSE140742 Nlrp3A350V",
        "healthy": "WT Control chow+placebo",
        "control": "Nlrp3 Control chow+placebo",
        "treatment": "GS-444217",
        "file_slug": "gse140742_nlrp3a350v",
    },
    "mouse_cdaa_lanifibranor": {
        "dataset_label": "GSE269493 CDAA-HFD",
        "healthy": "Chow",
        "control": "CDAA-HFD vehicle",
        "treatment": "Lanifibranor",
        "file_slug": "gse269493_cdaa_hfd",
    },
}


def parse_int_ranges(text):
    out = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, stop = chunk.split("-", 1)
            out.extend(range(int(start), int(stop) + 1))
        else:
            out.append(int(chunk))
    if not out:
        raise ValueError("No ensemble IDs were parsed.")
    return sorted(dict.fromkeys(out))


def load_context(args):
    context = scoring.load_context(args)
    for cohort_name, subset in context["mouse_subsets"].items():
        if subset.empty:
            raise ValueError(
                f"No mouse samples selected for {cohort_name}. Check the GEO accession "
                "and characteristics_ch1 patterns used by the final expression-mean "
                "scoring helper functions."
            )
    context["human_genes"] = np.asarray(context["human_genes"]).astype(str)
    context["mouse_genes"] = np.asarray(context["mouse_genes"]).astype(str)
    return context


def average_translated_expression(args, context, device):
    ensemble_ids, _, translated = scoring.average_ensemble_expression(args, context, device)
    return ensemble_ids, translated


def normalize_gene_symbol(gene):
    return str(gene).strip().upper()


def expression_frame(expr, sample_ids, genes):
    gene_names = [normalize_gene_symbol(g) for g in genes]
    df = pd.DataFrame(expr, index=np.asarray(sample_ids).astype(str), columns=gene_names)
    df = df.loc[:, [g not in ("", "NAN", "NONE") for g in df.columns]]
    df = df.T.groupby(level=0).mean().T
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    df = df.fillna(df.median(axis=0))
    return df


def preprocess_expression(expr_df):
    # ARCHS4 inputs are already log1p + quantile-normalized before model
    # training, so translated decoder means are treated as processed expression.
    processed = expr_df.replace([np.inf, -np.inf], np.nan).copy()
    processed = processed.fillna(processed.median(axis=0)).fillna(0.0)
    sd = processed.std(axis=0, ddof=0).replace(0, np.nan)
    z_expr = (processed - processed.mean(axis=0)) / sd
    z_expr = z_expr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return processed.astype(np.float32), z_expr.astype(np.float32)


def sample_metadata(cohort_name, subset, ensemble_ids):
    spec = COHORT_SPECS[cohort_name]
    meta_cols = [
        "dataset",
        "mouse_model",
        "mouse_genotype",
        "mouse_treatment",
        "time_point",
        "characteristics_ch1",
        "source_name_ch1",
        "title",
    ]
    frame = subset[[c for c in meta_cols if c in subset.columns]].copy()
    frame.insert(0, "sample_id", subset.index.astype(str))
    frame.insert(1, "cohort", cohort_name)
    frame.insert(2, "dataset_label", spec["dataset_label"])
    frame.insert(3, "ensemble_ids", ",".join(str(x) for x in ensemble_ids))
    frame.insert(4, "n_ensemble_models", len(ensemble_ids))
    frame["mouse_treatment"] = frame["mouse_treatment"].replace(
        {"Control chow+placebo": "Nlrp3 Control chow+placebo"}
    )
    frame["comparison_role"] = np.select(
        [
            frame["mouse_treatment"].eq(spec["healthy"]),
            frame["mouse_treatment"].eq(spec["control"]),
            frame["mouse_treatment"].eq(spec["treatment"]),
        ],
        ["healthy_reference", "disease_control", "treated"],
        default="other",
    )
    return frame


def parse_gmt(path):
    gene_sets = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = {normalize_gene_symbol(g) for g in parts[2:] if g}
            if genes:
                gene_sets[name] = genes
    return gene_sets


def hallmark_from_gseapy():
    import gseapy as gp

    libraries = [
        "MSigDB_Hallmark_2020",
        "Hallmark_2020",
        "h.all.v2023.1.Hs.symbols",
        "h.all.v2022.1.Hs.symbols",
    ]
    last_error = None
    for library in libraries:
        try:
            sets = gp.get_library(name=library, organism="Human")
            return {
                name: {normalize_gene_symbol(g) for g in genes}
                for name, genes in sets.items()
            }
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not retrieve Hallmark gene sets with gseapy: {last_error}")


def load_hallmark_sets(args):
    if args.hallmark_gmt is not None:
        path = Path(args.hallmark_gmt)
        if path.exists():
            return parse_gmt(path)
        raise FileNotFoundError(f"--hallmark_gmt does not exist: {path}")

    try:
        return hallmark_from_gseapy()
    except Exception as exc:
        raise RuntimeError(
            "Unable to retrieve Hallmark gene sets with gseapy. Pass "
            "--hallmark_gmt with an MSigDB Hallmark GMT file. Error: "
            f"{exc}"
        ) from exc


def ssgsea_rank_scores(processed_expr, gene_sets, min_size=10, max_size=500, alpha=0.25):
    genes = np.asarray(processed_expr.columns)
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    n_genes = len(genes)
    set_indices = {}
    for name, gene_set in gene_sets.items():
        idx = np.asarray([gene_to_idx[g] for g in gene_set if g in gene_to_idx], dtype=np.int64)
        idx = np.unique(idx)
        if min_size <= len(idx) <= max_size and len(idx) < n_genes:
            set_indices[name] = idx

    if not set_indices:
        raise ValueError("No Hallmark gene sets passed the overlap size filters.")

    values = processed_expr.to_numpy(dtype=np.float64)
    ranks = np.empty_like(values, dtype=np.float64)
    for i in range(values.shape[0]):
        ranks[i, :] = stats.rankdata(values[i, :], method="average")

    scores = np.zeros((values.shape[0], len(set_indices)), dtype=np.float64)
    set_names = list(set_indices)

    for i in range(values.shape[0]):
        order = np.argsort(-values[i, :], kind="mergesort")
        ordered_ranks = ranks[i, order]
        for j, name in enumerate(set_names):
            idx = set_indices[name]
            hit = np.isin(order, idx, assume_unique=False)
            k = int(hit.sum())
            if k == 0 or k == n_genes:
                scores[i, j] = np.nan
                continue
            hit_weights = np.zeros(n_genes, dtype=np.float64)
            hit_weights[hit] = np.power(ordered_ranks[hit], alpha)
            hit_sum = hit_weights.sum()
            if hit_sum <= 0:
                scores[i, j] = np.nan
                continue
            miss_weights = (~hit).astype(np.float64) / float(n_genes - k)
            running = np.cumsum(hit_weights / hit_sum - miss_weights)
            scores[i, j] = running.sum() / float(n_genes)

    return pd.DataFrame(scores, index=processed_expr.index, columns=set_names).astype(np.float32)


def activity_to_long(activity, metadata, cohort_name, activity_type):
    long = activity.reset_index(names="sample_id").melt(
        id_vars="sample_id", var_name="feature", value_name="activity"
    )
    long.insert(1, "cohort", cohort_name)
    long.insert(2, "dataset_label", COHORT_SPECS[cohort_name]["dataset_label"])
    long.insert(3, "activity_type", activity_type)
    return long.merge(metadata, on="sample_id", how="left")


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full_like(p, np.nan, dtype=np.float64)
    mask = np.isfinite(p)
    if not mask.any():
        return out
    idx = np.where(mask)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    n = float(len(ranked))
    q = ranked * n / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[order] = np.minimum(q, 1.0)
    return out


def contrast_activity(activity, metadata, cohort_name, activity_type):
    spec = COHORT_SPECS[cohort_name]
    groups = metadata.set_index("sample_id")["mouse_treatment"]
    control_ids = groups.index[groups.eq(spec["control"])]
    treatment_ids = groups.index[groups.eq(spec["treatment"])]
    healthy_ids = groups.index[groups.eq(spec["healthy"])]

    rows = []
    for feature in activity.columns:
        control = activity.loc[activity.index.intersection(control_ids), feature].dropna().to_numpy(float)
        treated = activity.loc[activity.index.intersection(treatment_ids), feature].dropna().to_numpy(float)
        healthy = activity.loc[activity.index.intersection(healthy_ids), feature].dropna().to_numpy(float)
        if len(control) == 0 or len(treated) == 0:
            continue
        mean_control = float(np.mean(control))
        mean_treated = float(np.mean(treated))
        mean_healthy = float(np.mean(healthy)) if len(healthy) else np.nan
        delta = mean_treated - mean_control
        pooled_sd = np.sqrt((np.var(control, ddof=1) + np.var(treated, ddof=1)) / 2.0) if len(control) > 1 and len(treated) > 1 else np.nan
        effect_size = delta / pooled_sd if pooled_sd and np.isfinite(pooled_sd) and pooled_sd > 0 else np.nan
        try:
            _, t_p = stats.ttest_ind(treated, control, equal_var=False, nan_policy="omit")
        except Exception:
            t_p = np.nan
        try:
            _, mw_p = stats.mannwhitneyu(treated, control, alternative="two-sided")
        except Exception:
            mw_p = np.nan
        rows.append(
            {
                "cohort": cohort_name,
                "dataset_label": spec["dataset_label"],
                "activity_type": activity_type,
                "feature": feature,
                "healthy_group": spec["healthy"],
                "disease_control_group": spec["control"],
                "treated_group": spec["treatment"],
                "comparison": f"{spec['treatment']} vs {spec['control']}",
                "n_control": int(len(control)),
                "n_treated": int(len(treated)),
                "n_healthy": int(len(healthy)),
                "mean_healthy": mean_healthy,
                "mean_disease_control": mean_control,
                "mean_treated": mean_treated,
                "delta_treated_minus_disease": delta,
                "abs_delta": abs(delta),
                "cohens_d_treated_minus_disease": effect_size,
                "abs_cohens_d": abs(effect_size) if np.isfinite(effect_size) else np.nan,
                "welch_t_p_value": t_p,
                "mannwhitney_p_value": mw_p,
                "direction": "higher_in_treated" if delta > 0 else "lower_in_treated",
                "description": (
                    "Drug-vs-disease contrast on per-sample activity scores. "
                    f"Positive delta means higher activity in {spec['treatment']} than "
                    f"{spec['control']}."
                ),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["welch_t_fdr"] = bh_fdr(out["welch_t_p_value"])
        out["mannwhitney_fdr"] = bh_fdr(out["mannwhitney_p_value"])
        out = out.sort_values(["activity_type", "abs_cohens_d", "abs_delta"], ascending=[True, False, False])
    return out


def zscore_rows(df):
    values = df.to_numpy(dtype=np.float64)
    mean = np.nanmean(values, axis=1, keepdims=True)
    sd = np.nanstd(values, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return pd.DataFrame((values - mean) / sd, index=df.index, columns=df.columns)


def safe_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_").lower()


def plot_contrast_bars(contrast, out_dir, top_n):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for (cohort_name, activity_type), sub in contrast.groupby(["cohort", "activity_type"]):
        spec = COHORT_SPECS[cohort_name]
        top = sub.reindex(sub["abs_cohens_d"].fillna(sub["abs_delta"]).sort_values(ascending=False).index).head(top_n)
        if top.empty:
            continue
        top = top.iloc[::-1]
        colors = np.where(top["delta_treated_minus_disease"] >= 0, "#E15759", "#4C78A8")
        fig_h = max(4.2, 0.28 * len(top) + 1.6)
        fig, ax = plt.subplots(figsize=(8.4, fig_h))
        ax.barh(top["feature"], top["delta_treated_minus_disease"], color=colors, alpha=0.86)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlabel(f"Mean activity difference: {spec['treatment']} - {spec['control']}")
        ax.set_ylabel("")
        ax.set_title(f"{spec['dataset_label']} {activity_type}: top treatment differences")
        ax.grid(axis="x", alpha=0.22)
        fig.tight_layout()
        fig.savefig(out_dir / f"{spec['file_slug']}_{safe_filename(activity_type)}_top_contrast_barplot.png", dpi=300)
        plt.close(fig)


def plot_activity_heatmaps(activity_mats, metadata_by_cohort, contrast, out_dir, top_n):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    for (cohort_name, activity_type), activity in activity_mats.items():
        spec = COHORT_SPECS[cohort_name]
        sub_contrast = contrast.loc[
            (contrast["cohort"] == cohort_name) & (contrast["activity_type"] == activity_type)
        ].copy()
        if sub_contrast.empty:
            continue
        top_features = sub_contrast.sort_values(
            ["abs_cohens_d", "abs_delta"], ascending=[False, False]
        )["feature"].head(top_n).tolist()
        top_features = [f for f in top_features if f in activity.columns]
        if not top_features:
            continue
        meta = metadata_by_cohort[cohort_name].set_index("sample_id")
        order_levels = [spec["healthy"], spec["control"], spec["treatment"]]
        sample_order = (
            meta.assign(_order=meta["mouse_treatment"].map({g: i for i, g in enumerate(order_levels)}))
            .sort_values(["_order", "mouse_treatment"])
            .index.tolist()
        )
        sample_order = [s for s in sample_order if s in activity.index]
        mat = activity.loc[sample_order, top_features].T
        mat_z = zscore_rows(mat)
        col_colors = meta.loc[sample_order, "mouse_treatment"].map(
            {
                spec["healthy"]: "#59A14F",
                spec["control"]: "#4C78A8",
                spec["treatment"]: "#E15759",
            }
        )
        width = max(8.0, 0.25 * len(sample_order) + 2.0)
        height = max(5.0, 0.24 * len(top_features) + 1.8)
        grid = sns.clustermap(
            mat_z,
            row_cluster=True,
            col_cluster=False,
            col_colors=col_colors,
            cmap="vlag",
            center=0,
            xticklabels=False,
            yticklabels=True,
            figsize=(width, height),
            cbar_kws={"label": "Row z-score activity"},
        )
        grid.fig.suptitle(f"{spec['dataset_label']} {activity_type}: top treatment-difference activities", y=1.02)
        grid.fig.savefig(out_dir / f"{spec['file_slug']}_{safe_filename(activity_type)}_top_activity_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close(grid.fig)


def write_outputs(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    context = load_context(args)
    ensemble_ids, translated = average_translated_expression(args, context, device)

    hallmark_sets = load_hallmark_sets(args)

    metadata_by_cohort = {}
    activity_mats = {}
    all_long = []
    all_contrasts = []

    for cohort_name in MOUSE_COHORTS:
        subset = context["mouse_subsets"][cohort_name]
        meta = sample_metadata(cohort_name, subset, ensemble_ids)
        metadata_by_cohort[cohort_name] = meta
        meta.to_csv(args.out_dir / f"{cohort_name}_sample_metadata.csv", index=False)

        expr = expression_frame(translated[cohort_name], subset.index.to_numpy(), context["human_genes"])
        expr.to_csv(args.out_dir / f"{cohort_name}_translated_expression_ensemble_mean.csv.gz")
        processed_expr, _ = preprocess_expression(expr)

        hallmark = ssgsea_rank_scores(
            processed_expr,
            hallmark_sets,
            min_size=args.min_geneset_size,
            max_size=args.max_geneset_size,
            alpha=args.ssgsea_alpha,
        )
        activity_type = "hallmark_ssgsea"
        hallmark.to_csv(args.out_dir / f"{cohort_name}_{activity_type}_scores.csv")
        activity_mats[(cohort_name, activity_type)] = hallmark
        all_long.append(activity_to_long(hallmark, meta, cohort_name, activity_type))
        all_contrasts.append(contrast_activity(hallmark, meta, cohort_name, activity_type))

    long = pd.concat(all_long, ignore_index=True)
    long.to_csv(args.out_dir / "per_sample_hallmark_ssgsea_long.csv", index=False)
    contrast = pd.concat(all_contrasts, ignore_index=True)
    contrast.to_csv(args.out_dir / "hallmark_ssgsea_treatment_vs_disease_contrasts.csv", index=False)
    top = (
        contrast.sort_values(["cohort", "activity_type", "abs_cohens_d", "abs_delta"], ascending=[True, True, False, False])
        .groupby(["cohort", "activity_type"], group_keys=False)
        .head(args.top_n)
    )
    top.to_csv(args.out_dir / "hallmark_ssgsea_treatment_vs_disease_top_features.csv", index=False)

    if not args.skip_plots:
        plot_contrast_bars(contrast, args.out_dir, args.top_n)
        plot_activity_heatmaps(activity_mats, metadata_by_cohort, contrast, args.out_dir, args.top_n)

    print(f"Wrote Hallmark ssGSEA outputs to: {args.out_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_ids", default="0-9")
    parser.add_argument("--full_model_prefix", default="full_ensemble")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--splits_dir", type=Path, default=None)
    parser.add_argument("--preproc_dir", type=Path, default=None)
    parser.add_argument("--model_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--human_gse", default=HUMAN_GSE)
    parser.add_argument("--mouse_gs_gse", default=MOUSE_GS_GSE)
    parser.add_argument("--mouse_cdaa_gse", default=MOUSE_CDAA_GSE)
    parser.add_argument("--hallmark_gmt", type=Path, default=None)
    parser.add_argument("--min_geneset_size", type=int, default=10)
    parser.add_argument("--max_geneset_size", type=int, default=500)
    parser.add_argument("--ssgsea_alpha", type=float, default=0.25)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.splits_dir = args.splits_dir or args.data_dir / "splits"
    args.preproc_dir = args.preproc_dir or args.data_dir / "preprocessed"
    args.model_dir = args.model_dir or args.data_dir / "models"
    parse_int_ranges(args.ensemble_ids)
    return args


def main():
    write_outputs(parse_args())


if __name__ == "__main__":
    main()
