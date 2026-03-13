
================================================================================
ARCHS4 DATA ACCESS SUMMARY FOR CROSS-SPECIES TRANSLATION MODEL
================================================================================

KEY FINDINGS:
1. ARCHS4 provides uniformly processed RNA-seq data for human and mouse
2. Data is stored in HDF5 format (~55GB human, ~36GB mouse)
3. archs4py package (pip install archs4py) provides efficient access
4. Remote access exists but is too slow for large-scale ML - download locally

INSTALLATION:
    pip install archs4py

DATA DOWNLOAD:
    import archs4py as a4
    a4.download.counts("human", path="./data/", version="latest")
    a4.download.counts("mouse", path="./data/", version="latest")

LIVER SAMPLE IDENTIFICATION (for external test set):
    Human pattern: "liver|hepat|hepg2|huh7|hepatocyte|hepatic|hepatoma|hepatocellular|hep3b|skhep|cholangio|biliary"
    Mouse pattern: "liver|hepat|hepa1-6|aml12|hepatocyte|hepatic|hepatoma|kupffer|cholangio|biliary"

    # Search metadata (fast, doesn't load expression data)
    liver_meta = a4.meta.meta(
        h5_file,
        search_term=liver_pattern,
        meta_fields=["characteristics_ch1", "source_name_ch1", "title"],
        remove_sc=True  # Exclude single-cell samples
    )
    liver_ids = set(liver_meta.index.tolist())

10-FOLD CROSS-VALIDATION:
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    # Create folds separately for human and mouse
    # Each fold holds out ~10% of each species

EFFICIENT DATA LOADING:
    # Load by sample indices (memory efficient)
    with h5py.File(h5_file, "r") as f:
        expression = f["data/expression"][:, sample_indices]

    # Or use archs4py functions:
    counts = a4.data.index(h5_file, sample_indices)
    counts = a4.data.samples(h5_file, sample_ids)

NORMALIZATION (optional):
    normalized = a4.normalize(counts, method="log_quantile")
    filtered = a4.utils.filter_genes(counts, readThreshold=20, sampleThreshold=0.02)

FILES CREATED:
1. archs4_workflow.py - Complete workflow script with all functions
2. train_fold.py - Single fold training script for cluster jobs
3. train_job.sh - SLURM array job submission script

WORKFLOW:
1. Run archs4_workflow.py once to download data and create splits
2. Submit train_job.sh as array job (0-9) for 10-fold CV
3. After CV, run train_final_model() with all data for final model

KEY API FUNCTIONS:
- a4.download.counts(species, path, version) - Download H5 files
- a4.data.meta(file, search_term, ...) - Search & get expression by metadata
- a4.data.index(file, indices) - Get expression by sample indices
- a4.data.samples(file, sample_ids) - Get expression by GEO IDs
- a4.meta.meta(file, search_term, ...) - Search metadata only (fast)
- a4.meta.get_meta_sample_field(file, field) - Get specific metadata field
