# ARCHS4 Database and archs4py Python API Documentation

## Overview

ARCHS4 (All RNA-seq and ChIP-seq Sample and Signature Search) is a comprehensive resource providing access to uniformly processed gene and transcript counts from human and mouse RNA-seq experiments from GEO (Gene Expression Omnibus) and SRA (Sequence Read Archive).

- **Website**: https://archs4.org
- **GitHub (archs4py)**: https://github.com/MaayanLab/archs4py
- **Citation**: Lachmann A, Torre D, Keenan AB, et al. Massive mining of publicly available RNA-seq data from human and mouse. Nature Communications 9. Article number: 1366 (2018)

## Key Features
- Over 1.5 million RNA-seq samples (as of 2023)
- Uniform processing using Kallisto aligner
- Human samples aligned to GRCh38 (Ensembl 107)
- Mouse samples aligned to GRCm39 (Ensembl 107)
- Single-cell sample probability prediction for filtering
- Available in HDF5 (H5) format for efficient data access

---

## Installation

```bash
pip install archs4py
```

---

## Downloading Data Files

ARCHS4 data is stored in large HDF5 files (30GB+ for each species).

### Download human gene counts:
```python
import archs4py as a4

# Download latest human gene counts
file_path = a4.download.counts("human", path="", version="latest")

# Download specific version
file_path = a4.download.counts("human", path="", version="2.5")
```

### Download mouse gene counts:
```python
import archs4py as a4

# Download latest mouse gene counts
file_path = a4.download.counts("mouse", path="", version="latest")
```

### Available Versions:
- `latest` - Most recent version
- `2.5` - August 2024
- `2.4` - June 2024  
- `2.3` - March 2024
- `2.2` - May 2023
- `2.1.2` - Earlier version
- `1.11` - November 2021

### File sizes (as of version 2.5):
- Human gene counts: ~45 GB
- Mouse gene counts: ~36 GB
- Human transcript counts: ~136 GB
- Mouse transcript counts: ~84 GB

### Direct download URLs:
- Human (latest): https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.latest.h5
- Mouse (latest): https://s3.dev.maayanlab.cloud/archs4/files/mouse_gene_v2.latest.h5

---

## Listing H5 File Contents

```python
import archs4py as a4

file = "human_gene_v2.5.h5"
a4.ls(file)
```

This will display the hierarchical structure of the H5 file showing:
- `data/expression` - Gene expression count matrix
- `meta/genes` - Gene metadata (gene_symbol, ensembl_id, etc.)
- `meta/samples` - Sample metadata (all available fields)

---

## Available Metadata Fields

The H5 files contain extensive metadata under `/meta/samples/`. Key fields include:

### Sample Identification:
- `geo_accession` - GEO sample accession (e.g., GSM1234567)
- `series_id` - GEO series ID (e.g., GSE12345)
- `sample` - SRA sample ID (e.g., SRS1234567)
- `sra_id` - SRA run ID

### Tissue/Cell Type Information (KEY FOR FILTERING):
- `source_name_ch1` - Source name/tissue description (often contains tissue info)
- `characteristics_ch1` - Sample characteristics (key field for tissue/cell type)
- `title` - Sample title (may contain tissue information)
- `extract_protocol_ch1` - Extraction protocol details

### Additional Metadata:
- `singlecellprobability` - ML-predicted probability of being single-cell (0-1)
- `readsaligned` - Number of aligned reads
- `library_source` - Library source type
- `library_selection` - Library selection method
- `library_strategy` - Library strategy (e.g., RNA-Seq)
- `molecule` - Molecule type
- `platform_id` - Sequencing platform

---

## Data Access Functions

### 1. Extract Random Samples
```python
import archs4py as a4

file = "human_gene_v2.5.h5"

# Extract 100 random samples, remove single-cell
rand_counts = a4.data.rand(file, 100, remove_sc=True, seed=42)

# Returns: pandas DataFrame with genes as rows, samples as columns
```

### 2. Search by Metadata (KEY FUNCTION)
```python
import archs4py as a4

file = "human_gene_v2.5.h5"

# Search for liver-related samples (case-insensitive, supports regex)
liver_counts = a4.data.meta(
    file, 
    search_term="liver",  # or "hepatocyte" or "liver|hepato"
    meta_fields=["characteristics_ch1", "source_name_ch1", "title"],
    remove_sc=True  # Exclude single-cell samples
)
```

### 3. Extract by Sample IDs (GEO Accessions)
```python
import archs4py as a4

file = "human_gene_v2.5.h5"
sample_ids = ["GSM1234567", "GSM1234568", "GSM1234569"]
counts = a4.data.samples(file, sample_ids)
```

### 4. Extract by Index Position
```python
import archs4py as a4

file = "human_gene_v2.5.h5"
counts = a4.data.index(file, sample_idx=[0, 1, 2, 3, 4])
```

### 5. Extract by GEO Series
```python
import archs4py as a4

file = "human_gene_v2.5.h5"
counts = a4.data.series(file, "GSE12345")
```

---

## Metadata-Only Access

### Get all metadata for a sample field:
```python
import archs4py as a4

file = "human_gene_v2.5.h5"

# Get all source names (useful for understanding data distribution)
sources = a4.meta.get_meta_sample_field(file, "source_name_ch1")
# Returns list of all values for that field

# Get sample characteristics
characteristics = a4.meta.get_meta_sample_field(file, "characteristics_ch1")

# Get all metadata as dictionary
all_meta = a4.meta.get_meta(file)
# Returns dict with all fields as keys
```

### Search metadata and return metadata (not expression):
```python
import archs4py as a4

file = "human_gene_v2.5.h5"

# Search metadata and return metadata DataFrame
meta_df = a4.meta.meta(
    file,
    search_term="liver",
    meta_fields=["characteristics_ch1", "source_name_ch1", "title"],
    remove_sc=True
)
# Returns DataFrame with metadata fields as columns
```

---

## Filtering Liver-Related Samples

For identifying liver-related samples for an external test set, search these metadata fields:

```python
import archs4py as a4
import re

file = "human_gene_v2.5.h5"  # or mouse_gene_v2.5.h5

# Comprehensive liver search pattern
liver_pattern = r"liver|hepat|hep\s*g2|hepg2|huh7|huh-7|hepatocyte|hepatic"

# Search in key metadata fields
liver_data = a4.data.meta(
    file,
    search_term=liver_pattern,
    meta_fields=[
        "characteristics_ch1",
        "source_name_ch1", 
        "title",
        "extract_protocol_ch1"
    ],
    remove_sc=True
)

# Alternative: Get metadata first, then filter
meta_df = a4.meta.meta(
    file,
    search_term=liver_pattern,
    meta_fields=[
        "characteristics_ch1",
        "source_name_ch1",
        "title"
    ],
    remove_sc=True
)

# Get sample IDs for liver samples
liver_sample_ids = meta_df.index.tolist()
```

### Key liver-related search terms:
- Human: "liver", "hepatocyte", "hepatic", "HepG2", "Huh7", "HuH-7", "hepatoma", "hepatocellular"
- Mouse: "liver", "hepatocyte", "hepatic", "Hepa1-6", "AML12", "hepatoma"

---

## Data Normalization

```python
import archs4py as a4

file = "human_gene_v2.5.h5"
counts = a4.data.rand(file, 100)

# Quantile normalization with log transform (recommended)
normalized = a4.normalize(counts, method="log_quantile")

# Other methods:
# normalized = a4.normalize(counts, method="quantile")
# normalized = a4.normalize(counts, method="cpm")
# normalized = a4.normalize(counts, method="tmm")
```

---

## Gene Filtering

```python
import archs4py as a4

file = "human_gene_v2.5.h5"
counts = a4.data.rand(file, 100)

# Filter genes with low expression
filtered = a4.utils.filter_genes(
    counts,
    readThreshold=20,      # Minimum reads
    sampleThreshold=0.02,  # Fraction of samples needed
    deterministic=True,    # Reproducible filtering
    aggregate=True         # Aggregate duplicate genes
)
```

---

## Remote Data Access (Without Downloading Full File)

```python
import archs4py as a4

# Access data directly from S3 without downloading full file
url = "https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.5.h5"

# Get random samples
rand_counts = a4.data.rand_remote(url, 100, remove_sc=True)

# Search metadata remotely
liver_counts = a4.data.meta_remote(
    url,
    search_term="liver",
    meta_fields=["characteristics_ch1", "source_name_ch1", "title"],
    remove_sc=True
)
```

---

## Best Practices for Large-Scale ML Workflows

### 1. Pre-download the H5 files
For repeated access, download files locally rather than using remote access:
```python
# Download once
a4.download.counts("human", path="/data/archs4/", version="latest")
a4.download.counts("mouse", path="/data/archs4/", version="latest")
```

### 2. Memory Management
- Don't load all samples at once (>1.5M samples)
- Use index-based access for batched processing
- Filter samples using metadata search before loading expression data

### 3. Batch Processing Strategy
```python
import archs4py as a4
import numpy as np

file = "human_gene_v2.5.h5"

# First, get all sample indices (fast)
all_geo_ids = a4.meta.get_meta_sample_field(file, "geo_accession")

# Shuffle and split for cross-validation
indices = np.arange(len(all_geo_ids))
np.random.shuffle(indices)

# Process in batches
batch_size = 10000
for i in range(0, len(indices), batch_size):
    batch_idx = indices[i:i+batch_size].tolist()
    batch_counts = a4.data.index(file, batch_idx)
    # Process batch...
```

### 4. Identifying and Excluding Test Set (Liver Samples)
```python
import archs4py as a4
import pandas as pd

# For both species
for species, file in [("human", "human_gene_v2.5.h5"), 
                       ("mouse", "mouse_gene_v2.5.h5")]:

    # Get all sample IDs
    all_ids = a4.meta.get_meta_sample_field(file, "geo_accession")

    # Search for liver samples (metadata only - fast)
    liver_meta = a4.meta.meta(
        file,
        search_term="liver|hepatocyte|hepatic|hepg2|huh7",
        meta_fields=["characteristics_ch1", "source_name_ch1", "title"],
        remove_sc=True
    )
    liver_ids = set(liver_meta.index.tolist())

    # Training IDs exclude liver
    train_ids = [x for x in all_ids if x not in liver_ids]

    print(f"{species}: {len(all_ids)} total, {len(liver_ids)} liver, {len(train_ids)} training")
```

### 5. Cross-Validation Split
```python
import archs4py as a4
import numpy as np
from sklearn.model_selection import KFold

file = "human_gene_v2.5.h5"

# Get training IDs (excluding liver)
all_ids = a4.meta.get_meta_sample_field(file, "geo_accession")
# ... filter out liver ...

# Create 10-fold CV splits
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_ids)):
    train_sample_ids = [train_ids[i] for i in train_idx]
    val_sample_ids = [train_ids[i] for i in val_idx]

    # Load data for this fold
    # train_counts = a4.data.samples(file, train_sample_ids)
    # val_counts = a4.data.samples(file, val_sample_ids)
```

### 6. HDF5 File Structure
```
/data
  /expression     - (n_genes x n_samples) count matrix, uint32
/meta
  /genes
    /gene_symbol  - Gene symbols
    /ensembl_id   - Ensembl gene IDs
  /samples
    /geo_accession        - GEO sample IDs
    /series_id            - GEO series IDs
    /characteristics_ch1  - Sample characteristics (tissue, cell type, etc.)
    /source_name_ch1      - Source name/tissue
    /title               - Sample title
    /singlecellprobability - ML-predicted single-cell probability
    /readsaligned         - Number of aligned reads
    ... and more
```

### 7. Direct HDF5 Access (for custom operations)
```python
import h5py
import numpy as np
import pandas as pd

file = "human_gene_v2.5.h5"

with h5py.File(file, "r") as f:
    # Get gene names
    genes = [x.decode("UTF-8") for x in f["meta/genes/gene_symbol"][:]]

    # Get sample IDs
    samples = [x.decode("UTF-8") for x in f["meta/samples/geo_accession"][:]]

    # Get expression for specific samples (by index)
    idx = [0, 1, 2]
    expression = f["data/expression"][:, idx]

    # Get specific metadata field
    characteristics = [x.decode("UTF-8") for x in f["meta/samples/characteristics_ch1"][:]]
```

---

## Summary of Key Functions

| Function | Purpose |
|----------|---------|
| `a4.download.counts(species, path, version)` | Download H5 data files |
| `a4.ls(file)` | List H5 file contents |
| `a4.data.meta(file, search_term, ...)` | Search and extract by metadata |
| `a4.data.rand(file, n, ...)` | Extract random samples |
| `a4.data.index(file, sample_idx, ...)` | Extract by index |
| `a4.data.samples(file, sample_ids)` | Extract by GEO IDs |
| `a4.data.series(file, series_id)` | Extract by GEO series |
| `a4.meta.get_meta(file)` | Get all metadata as dict |
| `a4.meta.get_meta_sample_field(file, field)` | Get specific metadata field |
| `a4.meta.meta(file, search_term, ...)` | Search and return metadata only |
| `a4.normalize(counts, method)` | Normalize expression data |
| `a4.utils.filter_genes(counts, ...)` | Filter low-expression genes |
| `a4.versions()` | List available data versions |

---

## Important Considerations

1. **Single-cell filtering**: Use `remove_sc=True` to exclude samples with singlecellprobability > 0.5
2. **Memory**: Each species has hundreds of thousands of samples; batch processing recommended
3. **Metadata quality**: GEO metadata varies in quality; manual curation may be needed for some analyses
4. **Version consistency**: Use the same version for human and mouse for cross-species analysis
5. **Gene mapping**: Human uses GRCh38, Mouse uses GRCm39, both with Ensembl 107 annotation
