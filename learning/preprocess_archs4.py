# run once after archs4_workflow.py
from pathlib import Path
import h5py, json, numpy as np
from archs4_workflow import load_splits
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

DATA_DIR   = Path("../archs4")
SPLITS_DIR = DATA_DIR / "splits"
OUT_DIR    = DATA_DIR / "preprocessed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

READ_THRESHOLD   = 20        # matches a4.utils.filter_genes defaults
SAMPLE_FRAC      = 0.02
QUANTILE_REF_N   = 20_000    # subsample used to build the quantile reference

def _streaming_gene_filter(h5_path, sample_cols, read_threshold, sample_frac):
    """One pass over gene rows: keep gene g if
       (# samples with counts >= read_threshold) >= sample_frac * len(sample_cols).
       Works gene-chunk-at-a-time -> natural HDF5 chunk layout -> fast."""
    with h5py.File(h5_path, "r") as f:
        dset = f["data/expression"]          # (n_genes, n_samples_total)
        n_genes = dset.shape[0]
        min_samples = int(np.ceil(sample_frac * len(sample_cols)))
        keep = np.zeros(n_genes, dtype=bool)
        # read in gene-chunks
        GENE_CHUNK = 2048
        sample_cols_sorted = np.sort(np.asarray(sample_cols))
        for g0 in range(0, n_genes, GENE_CHUNK):
            g1 = min(g0 + GENE_CHUNK, n_genes)
            # gene-major read (fast), then subset to train columns
            block = dset[g0:g1, :][:, sample_cols_sorted]     # (gene_chunk, n_train)
            keep[g0:g1] = (block >= read_threshold).sum(axis=1) >= min_samples
        return keep

def _build_quantile_ref(h5_path, sample_cols, kept_gene_idx, n_ref=QUANTILE_REF_N, seed=42):
    """Build a single mean-rank reference vector for quantile normalization
       using a random subsample of training columns."""
    rng = np.random.default_rng(seed)
    n = min(n_ref, len(sample_cols))
    ref_cols = np.sort(rng.choice(sample_cols, size=n, replace=False))
    with h5py.File(h5_path, "r") as f:
        dset = f["data/expression"]
        # gene-major read, subset rows to kept genes
        # read rows once, subset columns
        data = dset[:, ref_cols][kept_gene_idx, :]    # (G_filtered, n_ref)
    # log1p then per-sample ranks -> mean across samples
    logd = np.log1p(data.astype(np.float32))
    # ranks along gene axis for each sample
    sorted_vals = np.sort(logd, axis=0)
    ref = sorted_vals.mean(axis=1).astype(np.float32)  # (G_filtered,)
    return ref

def _quantile_normalize_col(col_log, ref_sorted):
    """Map a single sample's log1p-expression vector onto the reference via ranks."""
    order = np.argsort(col_log, kind="stable")
    out = np.empty_like(col_log)
    out[order] = ref_sorted
    return out

def preprocess_species(species, h5_path, split, folds):
    # All samples we will ever need: train (fold-union) + external test
    train_cols = np.asarray(split["train_indices"], dtype=np.int64)
    test_cols  = np.asarray(split["test_indices"],  dtype=np.int64)

    print2log(f"[{species}] filtering genes on {len(train_cols):,} training samples…")
    kept_mask = _streaming_gene_filter(h5_path, train_cols,
                                       READ_THRESHOLD, SAMPLE_FRAC)
    kept_idx  = np.where(kept_mask)[0]
    print2log(f"[{species}] kept {kept_idx.size:,} genes")

    with h5py.File(h5_path, "r") as f:
        genes = np.array([g.decode() for g in f["meta/genes/symbol"][:]])[kept_idx]
    np.save(OUT_DIR / f"{species}_genes.npy", genes)

    print2log(f"[{species}] building quantile reference…")
    ref_sorted = _build_quantile_ref(h5_path, train_cols, kept_idx)
    np.save(OUT_DIR / f"{species}_quantile_ref.npy", ref_sorted)

    # Write train matrix sample-major
    _write_matrix(h5_path, train_cols, kept_idx, ref_sorted,
                  OUT_DIR / f"{species}_X.npy",
                  OUT_DIR / f"{species}_sample_ids.npy", species, "train")

    # Write external (liver) test matrix
    _write_matrix(h5_path, test_cols, kept_idx, ref_sorted,
                  OUT_DIR / f"{species}_test_X.npy",
                  OUT_DIR / f"{species}_test_sample_ids.npy", species, "test")

    # Translate fold indices from H5-sample-index space into
    # row-index space of the new sample-major .npy
    h5idx_to_row = {int(h5idx): i for i, h5idx in enumerate(train_cols.tolist())}
    for k, fold in enumerate(folds):
        tr = np.array([h5idx_to_row[i] for i in fold["train_indices"]], dtype=np.int64)
        va = np.array([h5idx_to_row[i] for i in fold["val_indices"]],   dtype=np.int64)
        np.save(OUT_DIR / f"{species}_fold{k}_train_idx.npy", tr)
        np.save(OUT_DIR / f"{species}_fold{k}_val_idx.npy",   va)

def _write_matrix(h5_path, cols, kept_idx, ref_sorted, out_mat, out_ids, species, tag):
    """Stream through samples in gene-chunk blocks, normalize, write sample-major."""
    cols_sorted = np.sort(cols)
    # inverse permutation so we can write rows in the original `cols` order if desired
    # (order doesn't matter as long as sample_ids aligns)
    n_samples = cols_sorted.size
    G = kept_idx.size
    # Pre-allocate on-disk memmap (sample-major)
    arr = np.lib.format.open_memmap(out_mat, mode="w+",
                                    dtype=np.float32, shape=(n_samples, G))
    with h5py.File(h5_path, "r") as f:
        dset = f["data/expression"]
        sample_ids = np.array([s.decode() for s in f["meta/samples/geo_accession"][:]])
        np.save(out_ids, sample_ids[cols_sorted])

        # read in column-chunks of manageable size (here normalization is per-column anyway)
        COL_CHUNK = 4096
        for c0 in range(0, n_samples, COL_CHUNK):
            c1 = min(c0 + COL_CHUNK, n_samples)
            block_cols = cols_sorted[c0:c1]
            # Gene-major read: dset[:, block_cols]  -> (n_genes_total, chunk)
            # This is the same access pattern you already have, BUT it's done
            # ONCE TOTAL instead of once per fold.
            block = dset[:, block_cols][kept_idx, :].astype(np.float32)  # (G, chunk)
            block = np.log1p(block)
            # Quantile normalize each column against ref_sorted
            # block shape: (G_filtered, chunk)
            # log1p already applied
            order = np.argsort(block, axis=0)                           # (G, chunk)
            ranks = np.argsort(order, axis=0)                           # rank of each element
            out_block = ref_sorted[ranks].astype(np.float32)            # (G, chunk)
            # transpose once, write rows contiguously
            arr[c0:c1, :] = out_block.T
            if c0 % (COL_CHUNK * 10) == 0:
                print2log(f"[{species}/{tag}] {c1}/{n_samples} samples")
    arr.flush()
    del arr

def main():
    human_split, mouse_split, human_folds, mouse_folds = load_splits(SPLITS_DIR)
    preprocess_species("human", str(DATA_DIR / "human_gene_v2.latest.h5"),
                       human_split, human_folds)
    preprocess_species("mouse", str(DATA_DIR / "mouse_gene_v2.latest.h5"),
                       mouse_split, mouse_folds)

if __name__ == "__main__":
    main()
