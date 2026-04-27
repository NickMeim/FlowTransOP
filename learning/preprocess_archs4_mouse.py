# preprocess_archs4_mouse.py — resumable, single-argsort, cached filter/ref.
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

READ_THRESHOLD   = 20
SAMPLE_FRAC      = 0.02
QUANTILE_REF_N   = 20_000
COL_CHUNK        = 8192          # was 4096

def _streaming_gene_filter(h5_path, sample_cols, read_threshold, sample_frac):
    with h5py.File(h5_path, "r") as f:
        dset = f["data/expression"]
        n_genes = dset.shape[0]
        min_samples = int(np.ceil(sample_frac * len(sample_cols)))
        keep = np.zeros(n_genes, dtype=bool)
        GENE_CHUNK = 2048
        sample_cols_sorted = np.sort(np.asarray(sample_cols))
        for g0 in range(0, n_genes, GENE_CHUNK):
            g1 = min(g0 + GENE_CHUNK, n_genes)
            block = dset[g0:g1, :][:, sample_cols_sorted]
            keep[g0:g1] = (block >= read_threshold).sum(axis=1) >= min_samples
        return keep

def _build_quantile_ref(h5_path, sample_cols, kept_gene_idx, n_ref=QUANTILE_REF_N, seed=42):
    rng = np.random.default_rng(seed)
    n = min(n_ref, len(sample_cols))
    ref_cols = np.sort(rng.choice(sample_cols, size=n, replace=False))
    with h5py.File(h5_path, "r") as f:
        dset = f["data/expression"]
        data = dset[:, ref_cols][kept_gene_idx, :]
    logd = np.log1p(data.astype(np.float32))
    sorted_vals = np.sort(logd, axis=0)
    return sorted_vals.mean(axis=1).astype(np.float32)

def _detect_resume_point(out_mat, n_samples):
    """If out_mat exists from a previous job but no .progress file is around,
       scan for the first all-zero row (untouched memmap content)."""
    if not Path(out_mat).exists():
        return 0
    arr = np.load(out_mat, mmap_mode='r')
    if arr.shape[0] != n_samples:
        return 0
    SCAN_CHUNK = 16384
    for c0 in range(0, arr.shape[0], SCAN_CHUNK):
        c1 = min(c0 + SCAN_CHUNK, arr.shape[0])
        # row sums; quantile-normalized rows are strictly > 0 in expectation
        sums = arr[c0:c1].sum(axis=1)
        zr = np.where(sums == 0.0)[0]
        if zr.size > 0:
            return c0 + int(zr[0])
    return arr.shape[0]

def preprocess_species(species, h5_path, split, folds):
    train_cols = np.asarray(split["train_indices"], dtype=np.int64)
    test_cols  = np.asarray(split["test_indices"],  dtype=np.int64)

    # ---- cached gene filter ----
    kept_idx_path = OUT_DIR / f"{species}_kept_idx.npy"
    genes_path    = OUT_DIR / f"{species}_genes.npy"
    if kept_idx_path.exists() and genes_path.exists():
        kept_idx = np.load(kept_idx_path)
        print2log(f"[{species}] using cached kept_idx ({kept_idx.size:,} genes)")
    else:
        print2log(f"[{species}] filtering genes on {len(train_cols):,} training samples…")
        kept_mask = _streaming_gene_filter(h5_path, train_cols, READ_THRESHOLD, SAMPLE_FRAC)
        kept_idx  = np.where(kept_mask)[0]
        print2log(f"[{species}] kept {kept_idx.size:,} genes")
        np.save(kept_idx_path, kept_idx)
        with h5py.File(h5_path, "r") as f:
            genes = np.array([g.decode() for g in f["meta/genes/symbol"][:]])[kept_idx]
        np.save(genes_path, genes)

    # ---- cached quantile reference ----
    ref_path = OUT_DIR / f"{species}_quantile_ref.npy"
    if ref_path.exists():
        ref_sorted = np.load(ref_path)
        print2log(f"[{species}] using cached quantile reference")
    else:
        print2log(f"[{species}] building quantile reference…")
        ref_sorted = _build_quantile_ref(h5_path, train_cols, kept_idx)
        np.save(ref_path, ref_sorted)

    # ---- write train then test (each is independently resumable) ----
    _write_matrix(h5_path, train_cols, kept_idx, ref_sorted,
                  OUT_DIR / f"{species}_X.npy",
                  OUT_DIR / f"{species}_sample_ids.npy", species, "train")
    _write_matrix(h5_path, test_cols, kept_idx, ref_sorted,
                  OUT_DIR / f"{species}_test_X.npy",
                  OUT_DIR / f"{species}_test_sample_ids.npy", species, "test")

    # ---- fold-index translation (cheap; recomputed unconditionally) ----
    h5idx_to_row = {int(h5idx): i for i, h5idx in enumerate(np.sort(train_cols).tolist())}
    for k, fold in enumerate(folds):
        tr = np.array([h5idx_to_row[i] for i in fold["train_indices"]], dtype=np.int64)
        va = np.array([h5idx_to_row[i] for i in fold["val_indices"]],   dtype=np.int64)
        np.save(OUT_DIR / f"{species}_fold{k}_train_idx.npy", tr)
        np.save(OUT_DIR / f"{species}_fold{k}_val_idx.npy",   va)

def _write_matrix(h5_path, cols, kept_idx, ref_sorted, out_mat, out_ids, species, tag):
    cols_sorted = np.sort(cols)
    n_samples = cols_sorted.size
    G = kept_idx.size

    progress_file = Path(str(out_mat) + ".progress")

    # Decide resume point
    if progress_file.exists():
        c0_start = int(progress_file.read_text().strip())
        print2log(f"[{species}/{tag}] resuming from progress file at {c0_start}/{n_samples}")
        arr = np.lib.format.open_memmap(out_mat, mode="r+")
    elif Path(out_mat).exists():
        c0_start = _detect_resume_point(out_mat, n_samples)
        if c0_start > 0:
            print2log(f"[{species}/{tag}] no progress file, but detected "
                      f"{c0_start}/{n_samples} rows already written — resuming")
            arr = np.lib.format.open_memmap(out_mat, mode="r+")
        else:
            Path(out_mat).unlink()
            arr = np.lib.format.open_memmap(out_mat, mode="w+",
                                            dtype=np.float32, shape=(n_samples, G))
    else:
        c0_start = 0
        arr = np.lib.format.open_memmap(out_mat, mode="w+",
                                        dtype=np.float32, shape=(n_samples, G))

    if arr.shape != (n_samples, G):
        raise RuntimeError(f"{out_mat} shape {arr.shape} != expected {(n_samples, G)}; "
                           "delete it to start over.")

    with h5py.File(h5_path, "r") as f:
        dset = f["data/expression"]
        if not Path(out_ids).exists():
            sample_ids = np.array([s.decode() for s in f["meta/samples/geo_accession"][:]])
            np.save(out_ids, sample_ids[cols_sorted])

        ref_col = ref_sorted[:, None].astype(np.float32)   # (G, 1) -> broadcasts to (G, C)
        for c0 in range(c0_start, n_samples, COL_CHUNK):
            c1 = min(c0 + COL_CHUNK, n_samples)
            block_cols = cols_sorted[c0:c1]
            block = dset[:, block_cols][kept_idx, :].astype(np.float32)   # (G, C)
            np.log1p(block, out=block)
            order = np.argsort(block, axis=0)                              # (G, C)
            # Single argsort: out[order[r,c], c] = ref_sorted[r]
            out_block = np.empty_like(block)
            np.put_along_axis(out_block, order, ref_col, axis=0)
            arr[c0:c1, :] = out_block.T
            arr.flush()
            progress_file.write_text(str(c1))
            if (c0 // COL_CHUNK) % 5 == 0:
                print2log(f"[{species}/{tag}] {c1}/{n_samples} samples")

    arr.flush()
    del arr
    if progress_file.exists():
        progress_file.unlink()

def main():
    _, mouse_split, _, mouse_folds = load_splits(SPLITS_DIR)
    preprocess_species("mouse", str(DATA_DIR / "mouse_gene_v2.latest.h5"),
                       mouse_split, mouse_folds)

if __name__ == "__main__":
    main()