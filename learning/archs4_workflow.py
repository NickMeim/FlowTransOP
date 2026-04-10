#!/usr/bin/env python3
"""
ARCHS4 Data Access and Processing Pipeline for Cross-Species Translation Model
================================================================================

This script provides efficient data access patterns for:
1. Downloading human and mouse transcriptomic data from ARCHS4
2. Filtering out liver-related samples for external test set
3. Creating 10-fold cross-validation splits
4. Batch loading for memory-efficient training

IMPORTANT: Files are large (Human: ~55GB, Mouse: ~36GB)
           Download once to shared storage before running training jobs
"""

import archs4py as a4
import numpy as np
import pandas as pd
import h5py
import json
from sklearn.model_selection import KFold
from pathlib import Path
from typing import List, Dict, Tuple, Set

# ============================================================================
# CONFIGURATION
# ============================================================================

# Adjust these paths for your cluster environment
DATA_DIR = Path("../archs4")  # Shared storage for downloaded files
SPLITS_DIR = Path("../archs4/splits")        # Where to save CV splits
RANDOM_SEED = 42
N_FOLDS = 10

# File names (will be downloaded if not present)
HUMAN_FILE = "human_gene_v2.latest.h5"
MOUSE_FILE = "mouse_gene_v2.latest.h5"

# Comprehensive liver-related search patterns (regex, case-insensitive)
# Covers: liver tissue, hepatocytes, liver cell lines, organoids, hepatic conditions
LIVER_PATTERNS = {
    "human": r"liver|hepat|hep\s*g2|hepg2|huh[-\s]?7|huh7|hepatocyte|hepatic|" + 
             r"hepatoma|hepatocellular|hepatoblastoma|hep\s*3b|hep3b|" +
             r"sk[-\s]?hep|skhep|plc[-/]?prf|cholangio|biliary|bile\s*duct",
    "mouse": r"liver|hepat|hepa[-\s]?1[-\s]?6|hepa1-6|aml[-\s]?12|aml12|" +
             r"hepatocyte|hepatic|hepatoma|hepatoblast|brl[-\s]?3a|brl3a|" +
             r"cholangio|biliary|bile\s*duct|kupffer"
}

# Metadata fields to search for liver-related terms
SEARCH_FIELDS = [
    "characteristics_ch1",
    "source_name_ch1", 
    "title",
    "extract_protocol_ch1"
]


# ============================================================================
# STEP 1: DATA DOWNLOAD
# ============================================================================

def download_data(data_dir: Path = DATA_DIR, version: str = "latest", 
                  force: bool = False) -> Tuple[Path, Path]:
    """
    Download ARCHS4 human and mouse gene count files.

    Args:
        data_dir: Directory to save files
        version: ARCHS4 version (use 'latest' or specific like '2.5')
        force: Re-download even if files exist

    Returns:
        Tuple of (human_path, mouse_path)
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    human_path = data_dir / f"human_gene_v2.{version}.h5"
    mouse_path = data_dir / f"mouse_gene_v2.{version}.h5"

    # Download human data
    if not human_path.exists() or force:
        print(f"Downloading human gene counts (~55 GB)...")
        print(f"  Destination: {human_path}")
        a4.download.counts("human", path=str(data_dir), version=version)
    else:
        print(f"Human data already exists: {human_path}")

    # Download mouse data
    if not mouse_path.exists() or force:
        print(f"Downloading mouse gene counts (~36 GB)...")
        print(f"  Destination: {mouse_path}")
        a4.download.counts("mouse", path=str(data_dir), version=version)
    else:
        print(f"Mouse data already exists: {mouse_path}")

    return human_path, mouse_path


# ============================================================================
# STEP 2: IDENTIFY LIVER SAMPLES (EXTERNAL TEST SET)
# ============================================================================

def get_all_sample_ids(h5_file: str, remove_sc: bool = True) -> Tuple[List[str], np.ndarray]:
    """
    Get all sample IDs and their indices from H5 file.

    Args:
        h5_file: Path to ARCHS4 H5 file
        remove_sc: Exclude single-cell samples (singlecellprobability > 0.5)

    Returns:
        Tuple of (sample_ids, indices)
    """
    with h5py.File(h5_file, "r") as f:
        all_ids = [x.decode("UTF-8") for x in f["meta/samples/geo_accession"][:]]

        if remove_sc and "meta/samples/singlecellprobability" in f:
            sc_probs = f["meta/samples/singlecellprobability"][:]
            mask = sc_probs <= 0.5
            indices = np.where(mask)[0]
            sample_ids = [all_ids[i] for i in indices]
        else:
            indices = np.arange(len(all_ids))
            sample_ids = all_ids

    return sample_ids, indices


def identify_liver_samples(h5_file: str, species: str, 
                          remove_sc: bool = True) -> Tuple[Set[str], pd.DataFrame]:
    """
    Identify liver-related samples using comprehensive metadata search.

    Args:
        h5_file: Path to ARCHS4 H5 file
        species: 'human' or 'mouse'
        remove_sc: Exclude single-cell samples

    Returns:
        Tuple of (liver_sample_ids, liver_metadata_df)
    """
    liver_pattern = LIVER_PATTERNS[species]

    # Search metadata for liver-related samples
    liver_meta = a4.meta.meta(
        h5_file,
        search_term=liver_pattern,
        meta_fields=SEARCH_FIELDS,
        remove_sc=remove_sc,
        silent=True
    )

    liver_ids = set(liver_meta.index.tolist())

    print(f"  Found {len(liver_ids):,} liver-related samples")

    return liver_ids, liver_meta


def create_train_test_split(h5_file: str, species: str, 
                           remove_sc: bool = True) -> Dict:
    """
    Split samples into training (non-liver) and external test (liver) sets.

    Args:
        h5_file: Path to ARCHS4 H5 file
        species: 'human' or 'mouse'
        remove_sc: Exclude single-cell samples

    Returns:
        Dictionary with sample splits and metadata
    """
    print(f"\nProcessing {species} samples...")

    # Get all sample IDs
    all_ids, all_indices = get_all_sample_ids(h5_file, remove_sc=remove_sc)
    print(f"  Total samples: {len(all_ids):,}")

    # Identify liver samples
    liver_ids, liver_meta = identify_liver_samples(h5_file, species, remove_sc)

    # Create ID to index mapping
    id_to_idx = {sid: idx for sid, idx in zip(all_ids, all_indices)}

    # Split into train and test
    train_ids = [sid for sid in all_ids if sid not in liver_ids]
    test_ids = [sid for sid in all_ids if sid in liver_ids]

    train_indices = [id_to_idx[sid] for sid in train_ids]
    test_indices = [id_to_idx[sid] for sid in test_ids]

    print(f"  Training samples (non-liver): {len(train_ids):,}")
    print(f"  External test samples (liver): {len(test_ids):,}")

    return {
        "species": species,
        "all_ids": all_ids,
        "all_indices": all_indices.tolist(),
        "train_ids": train_ids,
        "train_indices": train_indices,
        "test_ids": test_ids,
        "test_indices": test_indices,
        "liver_metadata": liver_meta
    }


# ============================================================================
# STEP 3: CREATE 10-FOLD CROSS-VALIDATION SPLITS
# ============================================================================

def create_cv_folds(train_ids: List[str], train_indices: List[int],
                   n_folds: int = N_FOLDS, seed: int = RANDOM_SEED) -> List[Dict]:
    """
    Create k-fold cross-validation splits.

    In each fold, 10% of training data is held out as validation.
    Folds are created separately for each species to ensure:
    - Each fold has ~10% of human AND ~10% of mouse samples hidden

    Args:
        train_ids: List of training sample GEO IDs
        train_indices: Corresponding H5 file indices
        n_folds: Number of folds (default 10)
        seed: Random seed for reproducibility

    Returns:
        List of fold dictionaries with train/val splits
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    train_ids_array = np.array(train_ids)
    train_indices_array = np.array(train_indices)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_ids)):
        fold = {
            "fold": fold_idx,
            "train_ids": train_ids_array[train_idx].tolist(),
            "train_indices": train_indices_array[train_idx].tolist(),
            "val_ids": train_ids_array[val_idx].tolist(),
            "val_indices": train_indices_array[val_idx].tolist(),
        }
        folds.append(fold)
        print(f"    Fold {fold_idx + 1}: {len(fold['train_ids']):,} train, "
              f"{len(fold['val_ids']):,} val ({100*len(fold['val_ids'])/len(train_ids):.1f}%)")

    return folds


# ============================================================================
# STEP 4: SAVE SPLITS FOR CLUSTER JOBS
# ============================================================================
def convert_to_json_serializable(obj):
    """Convert NumPy types to JSON-serializable Python types."""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]

def save_splits(human_split: Dict, mouse_split: Dict,
               human_folds: List[Dict], mouse_folds: List[Dict],
               output_dir: Path = SPLITS_DIR) -> None:
    """
    Save all splits to disk for use by training jobs.

    Creates:
    - splits/human_split.json: Human train/test split
    - splits/mouse_split.json: Mouse train/test split
    - splits/human_folds.json: Human CV folds
    - splits/mouse_folds.json: Mouse CV folds
    - splits/liver_metadata_human.csv: Metadata for liver samples
    - splits/liver_metadata_mouse.csv: Metadata for liver samples
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits (without large metadata DataFrame)
    for species, split in [("human", human_split), ("mouse", mouse_split)]:
        split_save = {k: v for k, v in split.items() if k != "liver_metadata"}
        # Convert NumPy types to JSON-serializable types
        split_save = convert_to_json_serializable(split_save)
        with open(output_dir / f"{species}_split.json", "w") as f:
            json.dump(split_save, f)

        # Save liver metadata separately
        split["liver_metadata"].to_csv(output_dir / f"liver_metadata_{species}.csv")

    # Save CV folds (convert NumPy types)
    with open(output_dir / "human_folds.json", "w") as f:
        json.dump(convert_to_json_serializable(human_folds), f)
    with open(output_dir / "mouse_folds.json", "w") as f:
        json.dump(convert_to_json_serializable(mouse_folds), f)

    print(f"\nSplits saved to: {output_dir}")


def load_splits(splits_dir: Path = SPLITS_DIR) -> Tuple[Dict, Dict, List, List]:
    """
    Load pre-computed splits from disk.
    """
    with open(splits_dir / "human_split.json") as f:
        human_split = json.load(f)
    with open(splits_dir / "mouse_split.json") as f:
        mouse_split = json.load(f)
    with open(splits_dir / "human_folds.json") as f:
        human_folds = json.load(f)
    with open(splits_dir / "mouse_folds.json") as f:
        mouse_folds = json.load(f)

    return human_split, mouse_split, human_folds, mouse_folds


# ============================================================================
# STEP 5: EFFICIENT DATA LOADING FOR TRAINING
# ============================================================================

class ARCHS4DataLoader:
    """
    Efficient data loader for ARCHS4 HDF5 files.

    Designed for:
    - Memory-efficient batch loading
    - Cross-species translation model training
    - Cluster job compatibility
    """

    def __init__(self, human_file: str, mouse_file: str,
                 normalize: bool = True, filter_genes: bool = True):
        """
        Args:
            human_file: Path to human H5 file
            mouse_file: Path to mouse H5 file
            normalize: Apply log-quantile normalization
            filter_genes: Filter low-expression genes
        """
        self.human_file = human_file
        self.mouse_file = mouse_file
        self.normalize = normalize
        self.filter_genes = filter_genes

        # Cache gene names
        self._load_gene_info()

    def _load_gene_info(self):
        """Load gene names from both species."""
        with h5py.File(self.human_file, "r") as f:
            self.human_genes = [x.decode("UTF-8") for x in f["meta/genes/symbol"][:]]
        with h5py.File(self.mouse_file, "r") as f:
            self.mouse_genes = [x.decode("UTF-8") for x in f["meta/genes/symbol"][:]]

        print(f"Human genes: {len(self.human_genes):,}")
        print(f"Mouse genes: {len(self.mouse_genes):,}")

    def load_by_indices(self, h5_file: str, indices: List[int], 
                       batch_size: int = 5000) -> pd.DataFrame:
        """
        Load expression data by sample indices (memory efficient).

        Args:
            h5_file: Path to H5 file
            indices: List of sample indices to load
            batch_size: Number of samples per batch

        Returns:
            DataFrame with genes as rows, samples as columns
        """
        all_batches = []

        with h5py.File(h5_file, "r") as f:
            genes = [x.decode("UTF-8") for x in f["meta/genes/symbol"][:]]
            sample_ids = [x.decode("UTF-8") for x in f["meta/samples/geo_accession"][:]]

            for i in range(0, len(indices), batch_size):
                batch_idx = sorted(indices[i:i+batch_size])
                batch_data = f["data/expression"][:, batch_idx]
                batch_samples = [sample_ids[j] for j in batch_idx]

                batch_df = pd.DataFrame(
                    batch_data,
                    index=genes,
                    columns=batch_samples
                )
                all_batches.append(batch_df)

        result = pd.concat(all_batches, axis=1)

        # Apply preprocessing if requested
        if self.filter_genes:
            result = a4.utils.filter_genes(
                result, 
                readThreshold=20,
                sampleThreshold=0.02,
                deterministic=True
            )

        if self.normalize:
            result = a4.normalize(result, method="log_quantile")

        return result

    def load_fold_data(self, fold: Dict, species: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and validation data for a specific fold.

        Args:
            fold: Fold dictionary with train_indices and val_indices
            species: 'human' or 'mouse'

        Returns:
            Tuple of (train_data, val_data) DataFrames
        """
        h5_file = self.human_file if species == "human" else self.mouse_file

        train_data = self.load_by_indices(h5_file, fold["train_indices"])
        val_data = self.load_by_indices(h5_file, fold["val_indices"])

        return train_data, val_data

    def load_external_test(self, split: Dict, species: str) -> pd.DataFrame:
        """
        Load external test set (liver samples).

        Args:
            split: Split dictionary with test_indices
            species: 'human' or 'mouse'

        Returns:
            DataFrame with test data
        """
        h5_file = self.human_file if species == "human" else self.mouse_file
        return self.load_by_indices(h5_file, split["test_indices"])


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution workflow.

    Run this once to:
    1. Download data (if needed)
    2. Create train/test splits
    3. Create CV folds
    4. Save everything for cluster jobs
    """
    print("="*60)
    print("ARCHS4 DATA PREPARATION FOR CROSS-SPECIES TRANSLATION MODEL")
    print("="*60)

    # Step 1: Download data
    human_path, mouse_path = download_data(DATA_DIR, version="latest")

    # Step 2: Create train/test splits (liver = external test)
    human_split = create_train_test_split(str(human_path), "human", remove_sc=True)
    mouse_split = create_train_test_split(str(mouse_path), "mouse", remove_sc=True)

    # Step 3: Create 10-fold CV splits
    print("\nCreating 10-fold cross-validation splits...")
    print("  Human folds:")
    human_folds = create_cv_folds(
        human_split["train_ids"], 
        human_split["train_indices"]
    )
    print("  Mouse folds:")
    mouse_folds = create_cv_folds(
        mouse_split["train_ids"],
        mouse_split["train_indices"]
    )

    # Step 4: Save everything
    save_splits(human_split, mouse_split, human_folds, mouse_folds)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Human samples: {len(human_split['all_ids']):,}")
    print(f"  - Training (non-liver): {len(human_split['train_ids']):,}")
    print(f"  - External test (liver): {len(human_split['test_ids']):,}")
    print(f"Mouse samples: {len(mouse_split['all_ids']):,}")
    print(f"  - Training (non-liver): {len(mouse_split['train_ids']):,}")
    print(f"  - External test (liver): {len(mouse_split['test_ids']):,}")
    print(f"\nCV folds: {N_FOLDS}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"\nData files:")
    print(f"  Human: {human_path}")
    print(f"  Mouse: {mouse_path}")
    print(f"\nSplit files saved to: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
