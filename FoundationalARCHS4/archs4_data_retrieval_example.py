#!/usr/bin/env python3
"""
ARCHS4 Data Retrieval Script for Cross-Species Translation Model

This script demonstrates how to:
1. Download human and mouse transcriptomic data from ARCHS4
2. Filter out liver-related samples for an external test set
3. Organize data for 10-fold cross-validation

Author: Generated for cross-species translation model workflow
"""

import archs4py as a4
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
import h5py
from collections import Counter

# Configuration
DATA_DIR = "/data/archs4"  # Adjust to your data directory
RANDOM_SEED = 42
N_FOLDS = 10

# Liver-related search patterns (regex, case-insensitive)
LIVER_PATTERN_HUMAN = r"liver|hepat|hep\s*g2|hepg2|huh7|huh-7|hepatocyte|hepatic|hepatoma|hepatocellular"
LIVER_PATTERN_MOUSE = r"liver|hepat|hepa1-6|aml12|hepatocyte|hepatic|hepatoma"


def download_archs4_data(data_dir=DATA_DIR, version="latest"):
    """Download human and mouse gene count files from ARCHS4."""
    os.makedirs(data_dir, exist_ok=True)

    print("Downloading human gene counts...")
    human_path = a4.download.counts("human", path=data_dir, version=version)

    print("\nDownloading mouse gene counts...")
    mouse_path = a4.download.counts("mouse", path=data_dir, version=version)

    return human_path, mouse_path


def list_metadata_fields(h5_file):
    """List all available metadata fields in an ARCHS4 H5 file."""
    print(f"\n=== Metadata fields in {h5_file} ===")
    a4.ls(h5_file)


def get_sample_counts(h5_file):
    """Get total number of samples in the file."""
    with h5py.File(h5_file, "r") as f:
        n_samples = f["meta/samples/geo_accession"].shape[0]
        n_genes = f["meta/genes/gene_symbol"].shape[0]
    return n_samples, n_genes


def identify_liver_samples(h5_file, liver_pattern, remove_sc=True):
    """
    Identify liver-related samples using metadata search.

    Args:
        h5_file: Path to ARCHS4 H5 file
        liver_pattern: Regex pattern for liver-related terms
        remove_sc: Whether to exclude single-cell samples

    Returns:
        set: Set of GEO accession IDs for liver samples
    """
    # Search metadata for liver-related samples
    liver_meta = a4.meta.meta(
        h5_file,
        search_term=liver_pattern,
        meta_fields=[
            "characteristics_ch1",
            "source_name_ch1", 
            "title",
            "extract_protocol_ch1"
        ],
        remove_sc=remove_sc,
        silent=True
    )

    liver_ids = set(liver_meta.index.tolist())
    return liver_ids, liver_meta


def get_all_sample_ids(h5_file, remove_sc=True):
    """
    Get all sample IDs from the H5 file.

    Args:
        h5_file: Path to ARCHS4 H5 file
        remove_sc: Whether to exclude single-cell samples

    Returns:
        list: List of GEO accession IDs
    """
    all_ids = a4.meta.get_meta_sample_field(h5_file, "geo_accession")

    if remove_sc:
        sc_probs = a4.meta.get_meta_sample_field(h5_file, "singlecellprobability")
        all_ids = [gid for gid, prob in zip(all_ids, sc_probs) if prob <= 0.5]

    return all_ids


def create_train_test_split(h5_file, liver_pattern, remove_sc=True):
    """
    Split data into training (non-liver) and test (liver) sets.

    Returns:
        dict with train_ids, test_ids, and metadata
    """
    # Get all sample IDs
    all_ids = get_all_sample_ids(h5_file, remove_sc=remove_sc)

    # Identify liver samples
    liver_ids, liver_meta = identify_liver_samples(h5_file, liver_pattern, remove_sc)

    # Create train/test split
    train_ids = [gid for gid in all_ids if gid not in liver_ids]
    test_ids = [gid for gid in all_ids if gid in liver_ids]

    return {
        "all_ids": all_ids,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "liver_metadata": liver_meta
    }


def create_cv_folds(train_ids, n_folds=N_FOLDS, seed=RANDOM_SEED):
    """
    Create k-fold cross-validation splits.

    Returns:
        list of tuples: (fold_train_ids, fold_val_ids) for each fold
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    train_ids_array = np.array(train_ids)

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_ids)):
        fold_train_ids = train_ids_array[train_idx].tolist()
        fold_val_ids = train_ids_array[val_idx].tolist()
        folds.append((fold_train_ids, fold_val_ids))

    return folds


def load_expression_batch(h5_file, sample_ids, batch_size=5000):
    """
    Load expression data in batches to manage memory.

    Args:
        h5_file: Path to ARCHS4 H5 file
        sample_ids: List of GEO accession IDs to load
        batch_size: Number of samples per batch

    Yields:
        pd.DataFrame: Expression data for each batch
    """
    for i in range(0, len(sample_ids), batch_size):
        batch_ids = sample_ids[i:i+batch_size]
        batch_data = a4.data.samples(h5_file, batch_ids, silent=True)
        yield batch_data


def analyze_metadata_distribution(h5_file, field="source_name_ch1", top_n=20):
    """
    Analyze the distribution of values in a metadata field.
    """
    values = a4.meta.get_meta_sample_field(h5_file, field)
    counts = Counter(values)

    print(f"\n=== Top {top_n} values in {field} ===")
    for value, count in counts.most_common(top_n):
        print(f"  {count:8d}  {value[:80]}")

    return counts


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    # Example paths (adjust to your setup)
    HUMAN_H5 = os.path.join(DATA_DIR, "human_gene_v2.latest.h5")
    MOUSE_H5 = os.path.join(DATA_DIR, "mouse_gene_v2.latest.h5")

    # Step 1: Download data (uncomment to run)
    # download_archs4_data(DATA_DIR, version="latest")

    print("=" * 60)
    print("ARCHS4 Data Processing for Cross-Species Translation Model")
    print("=" * 60)

    # Step 2: Explore data structure
    # list_metadata_fields(HUMAN_H5)

    # Step 3: Get data statistics
    # n_samples, n_genes = get_sample_counts(HUMAN_H5)
    # print(f"\nHuman: {n_samples:,} samples, {n_genes:,} genes")

    # Step 4: Create train/test split for human
    # human_split = create_train_test_split(HUMAN_H5, LIVER_PATTERN_HUMAN)
    # print(f"\nHuman samples:")
    # print(f"  Total: {len(human_split['all_ids']):,}")
    # print(f"  Training (non-liver): {len(human_split['train_ids']):,}")
    # print(f"  Test (liver): {len(human_split['test_ids']):,}")

    # Step 5: Create train/test split for mouse
    # mouse_split = create_train_test_split(MOUSE_H5, LIVER_PATTERN_MOUSE)
    # print(f"\nMouse samples:")
    # print(f"  Total: {len(mouse_split['all_ids']):,}")
    # print(f"  Training (non-liver): {len(mouse_split['train_ids']):,}")
    # print(f"  Test (liver): {len(mouse_split['test_ids']):,}")

    # Step 6: Create 10-fold CV splits
    # human_folds = create_cv_folds(human_split["train_ids"])
    # mouse_folds = create_cv_folds(mouse_split["train_ids"])

    # print(f"\n10-fold CV splits created:")
    # for i, (train, val) in enumerate(human_folds):
    #     print(f"  Fold {i+1}: {len(train):,} train, {len(val):,} val")

    # Step 7: Load expression data (example with small sample)
    # sample_data = a4.data.rand(HUMAN_H5, 100, remove_sc=True, seed=42)
    # print(f"\nSample data shape: {sample_data.shape}")
    # print(f"Genes: {sample_data.shape[0]}, Samples: {sample_data.shape[1]}")

    # Step 8: Normalize data
    # normalized = a4.normalize(sample_data, method="log_quantile")
    # print(f"\nNormalized data shape: {normalized.shape}")

    # Step 9: Filter low-expression genes
    # filtered = a4.utils.filter_genes(
    #     sample_data,
    #     readThreshold=20,
    #     sampleThreshold=0.02
    # )
    # print(f"\nFiltered data shape: {filtered.shape}")

    print("\nScript complete. Uncomment sections to run specific analyses.")
