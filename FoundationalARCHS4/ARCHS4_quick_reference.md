# ARCHS4 + archs4py Quick Reference Card

## Installation
```bash
pip install archs4py
```

## Basic Import
```python
import archs4py as a4
```

## Download Data Files (~30-45 GB each)
```python
# Human gene counts
a4.download.counts("human", path="./data/", version="latest")

# Mouse gene counts
a4.download.counts("mouse", path="./data/", version="latest")
```

## List H5 File Structure
```python
a4.ls("human_gene_v2.latest.h5")
```

## Get Available Versions
```python
a4.versions()
# Returns: ['latest', '2.5', '2.4', '2.3', '2.2', '2.1.2', '1.11']
```

---

## DATA ACCESS FUNCTIONS

### Random Samples
```python
counts = a4.data.rand(file, 1000, remove_sc=True, seed=42)
```

### Search by Metadata (KEY FOR TISSUE FILTERING)
```python
# Search for specific tissue/condition
counts = a4.data.meta(
    file, 
    search_term="liver|hepatocyte",  # regex supported
    meta_fields=["characteristics_ch1", "source_name_ch1", "title"],
    remove_sc=True
)
```

### By GEO Sample IDs
```python
counts = a4.data.samples(file, ["GSM1234567", "GSM1234568"])
```

### By Index Position
```python
counts = a4.data.index(file, [0, 1, 2, 3, 4])
```

### By GEO Series
```python
counts = a4.data.series(file, "GSE12345")
```

---

## METADATA FUNCTIONS

### Get All Metadata as Dict
```python
meta_dict = a4.meta.get_meta(file)
```

### Get Specific Metadata Field
```python
# Useful for batch filtering
sources = a4.meta.get_meta_sample_field(file, "source_name_ch1")
sample_ids = a4.meta.get_meta_sample_field(file, "geo_accession")
sc_probs = a4.meta.get_meta_sample_field(file, "singlecellprobability")
```

### Search Metadata (Returns Metadata, Not Expression)
```python
meta_df = a4.meta.meta(file, "liver", remove_sc=True)
# Returns DataFrame with metadata columns
```

---

## KEY METADATA FIELDS FOR FILTERING

| Field | Description |
|-------|-------------|
| `geo_accession` | GEO sample ID (e.g., GSM1234567) |
| `series_id` | GEO series ID (e.g., GSE12345) |
| `characteristics_ch1` | **KEY: Sample characteristics (tissue, cell type)** |
| `source_name_ch1` | **KEY: Source tissue/cell name** |
| `title` | Sample title (may contain tissue info) |
| `singlecellprobability` | ML-predicted single-cell probability (0-1) |
| `readsaligned` | Number of aligned reads |

---

## NORMALIZATION

```python
# Log-quantile normalization (recommended)
normalized = a4.normalize(counts, method="log_quantile")

# Other methods: "quantile", "cpm", "tmm"
```

---

## GENE FILTERING

```python
filtered = a4.utils.filter_genes(
    counts,
    readThreshold=20,      # Min reads per gene
    sampleThreshold=0.02,  # Fraction of samples needed
    aggregate=True         # Sum duplicate genes
)
```

---

## LIVER SAMPLE IDENTIFICATION

### Human liver patterns:
```python
"liver|hepat|hepg2|huh7|hepatocyte|hepatic|hepatoma|hepatocellular"
```

### Mouse liver patterns:
```python
"liver|hepat|hepa1-6|aml12|hepatocyte|hepatic"
```

### Example workflow:
```python
# Get liver sample IDs
liver_meta = a4.meta.meta(
    file,
    "liver|hepatocyte|hepatic",
    meta_fields=["characteristics_ch1", "source_name_ch1", "title"],
    remove_sc=True
)
liver_ids = set(liver_meta.index.tolist())

# Get all sample IDs
all_ids = a4.meta.get_meta_sample_field(file, "geo_accession")

# Training = non-liver
train_ids = [x for x in all_ids if x not in liver_ids]
```

---

## REMOTE ACCESS (No Download Required)

```python
url = "https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.latest.h5"

# Random samples from remote
counts = a4.data.rand_remote(url, 100, remove_sc=True)

# Metadata search from remote
counts = a4.data.meta_remote(url, "liver", remove_sc=True)
```

---

## DIRECT HDF5 ACCESS

```python
import h5py

with h5py.File(file, "r") as f:
    # Gene names
    genes = [x.decode("UTF-8") for x in f["meta/genes/gene_symbol"][:]]

    # Sample IDs  
    samples = [x.decode("UTF-8") for x in f["meta/samples/geo_accession"][:]]

    # Expression by index
    expression = f["data/expression"][:, [0, 1, 2]]

    # Get shape
    n_genes, n_samples = f["data/expression"].shape
```

---

## DATA FILE SIZES (v2.5)

| File | Size |
|------|------|
| Human gene counts | ~45 GB |
| Mouse gene counts | ~36 GB |
| Human transcript counts | ~136 GB |
| Mouse transcript counts | ~84 GB |

---

## IMPORTANT NOTES

1. **remove_sc=True**: Excludes samples with singlecellprobability > 0.5
2. **Search is case-insensitive and supports regex**
3. **HDF5 format allows efficient sliced access without loading full file**
4. **Gene symbols use Ensembl 107 annotation**
5. **Human: GRCh38, Mouse: GRCm39 reference genomes**
