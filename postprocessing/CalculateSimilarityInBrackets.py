#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info


def _pair_key_and_side(stem: str):
    """
    Given a file stem (without .npy), infer pair key and whether it is set 1 or 2.
    Returns (key, side) where side is 1 or 2, or (None, None) if pattern not recognized.

    Examples
    --------
    best_genes_1                    -> key='best_genes_X', side=1
    best_genes_2                    -> key='best_genes_X', side=2
    high_correlation_set_1_iter0    -> key='high_correlation_set_X_iter0', side=1
    high_correlation_set_2_iter0    -> key='high_correlation_set_X_iter0', side=2
    """
    if '_1_' in stem:
        return stem.replace('_1_', '_X_'), 1
    if stem.endswith('_1'):
        return stem[:-2] + '_X', 1
    if '_2_' in stem:
        return stem.replace('_2_', '_X_'), 2
    if stem.endswith('_2'):
        return stem[:-2] + '_X', 2
    return None, None


def _base_label_from_key(key: str) -> str:
    """
    Turn something like:
      'best_genes_X'               -> 'best_genes'
      'high_correlation_set_X_iter0' -> 'high_correlation_set_iter0'
    """
    if key.endswith('_X'):
        return key[:-2]
    return key.replace('_X_', '_')


def _iteration_from_base_label(base_label: str):
    """
    Map subset base_label to iteration index, using user-provided mapping:

      high_correlation_set_iter0 -> 0
      high_correlation_set_iter1 -> 1
      high_correlation_set_iter2 -> 2
      high_correlation_set_iter3 -> 3
      high_correlation_set_iter4 -> 4
      best_genes                 -> 5
      second_best_genes          -> 6
      third_best_genes           -> 7
      fourth_best_genes          -> 8
    """
    # High-correlation sets
    if base_label.startswith("high_correlation_set_iter"):
        suffix = base_label.replace("high_correlation_set_iter", "")
        try:
            return int(suffix)
        except ValueError:
            return None

    # Bracketed best sets
    if base_label == "best_genes":
        return 5
    if base_label == "second_best_genes":
        return 6
    if base_label == "third_best_genes":
        return 7
    if base_label == "fourth_best_genes":
        return 8

    # Anything else we don't know how to map
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean cross-correlation of predefined gene subsets for each cell line."
    )
    parser.add_argument(
        '--cell_lines', metavar='N', type=str, nargs='*',
        default=[
            "PC3", "HT29", "MCF7", "A549", "NPC", "HEPG2", "A375", "YAPC",
            "U2OS", "MCF10A", "HA1E", "HCC515", "ASC", "VCAP", "HUVEC", "HELA"
        ],
        help='Cell lines to use (must match folder names under cell_data_root).'
    )
    parser.add_argument(
        '--cell_data_root', type=str,
        default='../preprocessing/preprocessed_data/SameCellimputationModel/',
        help='Root directory with per-cell train_*.csv and val_*.csv splits.'
    )
    parser.add_argument(
        '--subset_root', type=str,
        default='../preprocessing/preprocessed_data/SameCellimputationModel/bracketed_difficulty/',
        help='Directory where the gene subset .npy files live (brackets + best_plus_same_features).'
    )
    parser.add_argument(
        '--cmap_file', type=str,
        default='../../TranslationalModels/OmicTranslationBenchmark/preprocessing/preprocessed_data/cmap_all_genes_q1_tas03.csv',
        help='Path to the CMAP expression matrix used to derive the gene subsets.'
    )
    parser.add_argument(
        '--fold_id', type=int, default=0,
        help='Which fold to use; train_<fold_id>.csv and val_<fold_id>.csv will be combined.'
    )
    parser.add_argument(
        '--output_csv', type=str, default='MeanSubsetCorrelationsWithIterations.csv',
        help='Output CSV file name.'
    )

    args = parser.parse_args()

    subset_root = Path(args.subset_root)
    cell_data_root = Path(args.cell_data_root)

    # -------------------------------------------------------------------------
    # 1. Discover all .npy gene subset files (root + best_plus_same_features)
    #    and organize them into (set1, set2) pairs
    # -------------------------------------------------------------------------
    pairs = {}  # key -> {1: Path, 2: Path}

    # .npy directly in subset_root (bracketed best/second/third/fourth sets)
    npy_paths = list(subset_root.glob('*.npy'))

    # .npy inside best_plus_same_features (high-correlation sets)
    best_plus_dir = subset_root / 'best_plus_same_features'
    if best_plus_dir.exists():
        npy_paths.extend(best_plus_dir.glob('*.npy'))

    for npy_path in sorted(npy_paths):
        stem = npy_path.stem  # filename without .npy
        key, side = _pair_key_and_side(stem)
        if key is None:
            continue
        if key not in pairs:
            pairs[key] = {}
        pairs[key][side] = npy_path

    # Keep only complete pairs (we need both 1 and 2)
    complete_pairs = {k: v for k, v in pairs.items() if 1 in v and 2 in v}
    if not complete_pairs:
        raise RuntimeError(
            f"No complete (1/2) gene subset pairs found under {subset_root} "
            f"(including best_plus_same_features)."
        )

    print2log("Found the following gene subset pairs:")
    for key, sides in complete_pairs.items():
        print2log(f"  {key}: {sides[1].name}, {sides[2].name}")

    # -------------------------------------------------------------------------
    # 2. Load expression matrix once
    # -------------------------------------------------------------------------
    print2log(f"Loading expression matrix from {args.cmap_file}")
    cmap_all = pd.read_csv(args.cmap_file, index_col=0)

    # -------------------------------------------------------------------------
    # 3. Pre-load all gene sets, record base_label & iteration, and build global union of genes
    # -------------------------------------------------------------------------
    subset_info = []  # list of dicts: {key, base_label, iteration, genes1, genes2}
    union_genes = set()

    for key, sides in complete_pairs.items():
        base_label = _base_label_from_key(key)
        iteration = _iteration_from_base_label(base_label)
        if iteration is None:
            print2log(f"Skipping subset pair with base_label '{base_label}' (no iteration mapping).")
            continue

        genes1 = np.load(sides[1], allow_pickle=True)
        genes2 = np.load(sides[2], allow_pickle=True)
        genes1 = np.array(genes1, dtype=str)
        genes2 = np.array(genes2, dtype=str)

        subset_info.append({
            "key": key,
            "base_label": base_label,
            "iteration": iteration,
            "genes1": genes1,
            "genes2": genes2,
        })
        union_genes.update(genes1)
        union_genes.update(genes2)

    if not subset_info:
        raise RuntimeError("No subset pairs with a valid iteration mapping were found.")

    # Restrict to genes that are in the expression matrix
    union_genes = sorted(set(cmap_all.columns).intersection(union_genes))
    if not union_genes:
        raise RuntimeError("None of the genes from the subset .npy files are present in the expression matrix.")

    print2log(f"Total unique genes across all subsets (after intersecting with cmap): {len(union_genes)}")

    # -------------------------------------------------------------------------
    # 4. For each cell line: use fold_0 train+val sigs (all data), compute correlation matrix
    #    over union_genes, then compute mean cross-correlation for each subset pair
    # -------------------------------------------------------------------------
    results = []

    for cell in args.cell_lines:
        print2log(f"\nProcessing cell line: {cell}")
        cell_dir = cell_data_root / cell

        train_path = cell_dir / f"train_{args.fold_id}.csv"
        val_path = cell_dir / f"val_{args.fold_id}.csv"

        if not train_path.exists() or not val_path.exists():
            print2log(f"  WARNING: Missing train/val CSV for cell {cell} at fold {args.fold_id}; skipping cell.")
            continue

        train_info = pd.read_csv(train_path, index_col=0)
        val_info = pd.read_csv(val_path, index_col=0)

        if 'sig_id' not in train_info.columns or 'sig_id' not in val_info.columns:
            raise ValueError(f"'sig_id' column missing in {train_path} or {val_path}")

        sig_ids = sorted(set(train_info['sig_id']).union(val_info['sig_id']))

        # Restrict to signatures present in cmap
        sig_ids = [s for s in sig_ids if s in cmap_all.index]
        if not sig_ids:
            print2log(f"  WARNING: No matching signatures found in cmap for cell {cell}; skipping cell.")
            continue

        # All data for this cell (train+val of fold 0)
        cmap_cell = cmap_all.loc[sig_ids, union_genes]

        # Gene-gene correlation matrix for this cell across all union_genes
        r_mat = np.corrcoef(cmap_cell.values.T)
        r_df = pd.DataFrame(r_mat, index=union_genes, columns=union_genes)

        for info in subset_info:
            base_label = info["base_label"]
            iteration = info["iteration"]

            # Intersect subset gene lists with available genes
            g1 = [g for g in info["genes1"] if g in union_genes]
            g2 = [g for g in info["genes2"] if g in union_genes]

            if len(g1) == 0 or len(g2) == 0:
                mean_r = np.nan
                print2log(
                    f"  Pair {base_label} (iteration {iteration}): empty intersection for cell {cell} "
                    f"(g1={len(g1)}, g2={len(g2)})."
                )
            else:
                r_sub = r_df.loc[g1, g2]
                mean_r = float(np.nanmean(r_sub.values))
                print2log(
                    f"  Pair {base_label} (iteration {iteration}): "
                    f"mean cross-correlation for cell {cell} = {mean_r:.4f}"
                )

            # === Final output row ===
            results.append({
                "iteration": iteration,
                "correlation": mean_r,
                "cell": cell,
            })

    if not results:
        print2log("No results were computed; nothing to save.")
        return

    df_out = pd.DataFrame(results)
    df_out.to_csv(args.output_csv, index=False)
    print2log(f"\nSaved mean subset correlations (with iterations & cells) to {args.output_csv}")


if __name__ == "__main__":
    main()
