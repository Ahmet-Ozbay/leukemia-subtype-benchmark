# leukemia-subtype-benchmark
End-to-end, leak-free machine learning benchmark for ALL vs. AML leukemia subtype classification on high-dimensional microarray gene expression data.

## Setup Instructions
You can activate and verify the setup with either Conda or venv.

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate leukemia-subtype-benchmark
```

### Using Standard Python venv & pip
```bash
python3.11 -m venv .venv
source .venv/bin/activate  

# On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Verify the Environment

```bash
python -c "import sklearn, shap, umap, optuna, GEOparse; print('Environment successfully initialized')"
```

## Data Pipeline & Preprocessing

The benchmark evaluates models on leukemia microarray cohorts framed strictly as a binary subtype classification problem (**ALL vs. AML**):
- **Golub et al. (1999):** Primary benchmark ($n=72$, $p=7{,}129$). Staged via Kaggle Hub.
- **Armstrong et al. (2002) / GSE28497:** External validation cohort from NCBI GEO.
- **MILE Study / GSE13159:** Large-scale cross-study cohort from NCBI GEO.

### 1. Automated Data Acquisition

Datasets can be acquired programmatically using the download pipeline without manually downloading or placing files:

```bash
# Download all cohorts (Golub, Armstrong, MILE)
python -m src.data.download_data

# Or download a specific dataset:
python -m src.data.download_data --dataset golub
python -m src.data.download_data --dataset armstrong
python -m src.data.download_data --dataset mile
```

> [!NOTE]
> **Kaggle API Token (Golub Dataset):**
> Acquiring the Golub dataset requires a Kaggle API token. You can configure it interactively by running:
> ```bash
> python scripts/setup_kaggle_auth.py
> ```
> Alternatively, set `KAGGLE_API_TOKEN` in a `.env` file at the project root or in your shell environment. If not detected when running the download script, you will be prompted to configure it.
> GEO datasets (*Armstrong* and *MILE*) are retrieved publicly via NCBI GEO without authentication.

Downloaded raw archives and tables are automatically organized into dedicated directories under `data/raw/`:
- `data/raw/golub/`
- `data/raw/armstrong_2002/`
- `data/raw/mile_study/`

### 2. Preprocessing Protocol

Once raw data files are acquired, the ingestion pipeline standardizes and sanitizes each cohort:
1. **Matrix Transposition:** Converts raw gene-by-sample matrices into standard sample-by-feature layout ($n=72$, $p=7{,}129$).
2. **Metadata Sanitization:** Drops non-numeric call confidence flags (`P`, `A`, `M`) and aligns patient identifiers against ground truth labels without relying on column ordering.
3. **Thresholding & Imputation:** Clips expression intensities to $[20.0, 16000.0]$ to suppress background noise and sensor saturation, followed by median imputation for any remaining missing values.
4. **Serialization:** Exports cleaned probe-level partitions to `data/processed/probes/` as high-performance Parquet files.

#### Running Preprocessing

To execute data transformation and preprocessing:

```bash
# Process all downloaded cohorts
python -m src.data.make_dataset

# Or process a specific cohort:
python -m src.data.make_dataset --dataset golub
python -m src.data.make_dataset --dataset armstrong
```

### 3. Probe-to-Gene Mapping & Cross-Platform Harmonization

Because different microarray platforms use distinct probe identifier systems, cross-cohort validation requires mapping probes to standardized **HUGO gene symbols** and harmonizing overlapping feature spaces:
- **Golub et al. (1999):** Affymetrix HuGeneFL / Hu6800 (GEO GPL80) $\to$ 7,129 probes mapped to 5,545 unique HUGO symbols.
- **Armstrong et al. (2002) / GSE28497:** Affymetrix Human Genome U133A (GEO GPL96) $\to$ 22,283 probes mapped to 13,237 unique HUGO symbols.
- **Biomarker Validation:** 100% verified cross-platform match for key leukemia biomarkers (*CST3*, *CD33*, *CD19*, *MPO*, *FAH*, *MME/CD10*, *LYZ*).
- **Cross-Platform Intersection:** **5,426 shared HUGO genes** (97.9% coverage of Golub's mapped genes).

#### Zero-Leakage Probe Collapsing
When multiple probes target the same gene, `ProbeCollapser` (`sklearn.base.BaseEstimator`, `TransformerMixin`) aggregates them into a single gene feature. To prevent data leakage:
- When using `max_variance`, probe variances are computed **strictly on training fold samples** inside the pipeline.
- On benchmark splits, probe selection fits strictly on `golub_train_38` and transforms `golub_test_34`.
- Row-wise `mean` and `median` strategies are also supported.

#### Running Gene Mapping & Harmonization

```bash
# Extract annotations and generate harmonization / biomarker summary tables:
python -m src.data.map_genes

# Full pipeline: extract annotations, collapse to genes, and harmonize cohorts:
python -m src.data.map_genes --collapse --harmonize

# Custom strategy (e.g. drop multi-mapped probes, use mean aggregation):
python -m src.data.map_genes --collapse --harmonize --drop-multi-mapped --strategy mean
```

Generated reports are persisted to:
- `outputs/tables/gene_harmonization_summary.md` (and `.csv`)
- `outputs/tables/biomarker_probe_mapping.md` (and `.csv`)

### 4. Processed Directory Organization & Data Accessor API

All processed matrices are organized under `data/processed/` by feature abstraction level:

```
data/processed/
├── probes/               # Level 1: Quality-controlled probe-level matrices (p >> n)
│   ├── golub_train_38.parquet
│   ├── golub_test_34.parquet
│   ├── golub_full_72.parquet
│   └── armstrong_full.parquet
│
├── genes/                # Level 2: Collapsed HUGO gene-level matrices
│   ├── golub_train_genes.parquet
│   ├── golub_test_genes.parquet
│   ├── golub_full_genes.parquet
│   └── armstrong_full_genes.parquet
│
├── harmonized/           # Level 3: Cross-platform harmonized matrices (5426 shared genes)
│   ├── golub_harmonized_genes.parquet
│   └── armstrong_harmonized_genes.parquet
│
└── annotations/          # Platform probe-to-gene lookup tables (.parquet & .csv)
    ├── golub_probe_to_gene.parquet
    └── armstrong_probe_to_gene.parquet
```

#### Unified `load_dataset` Helper

Downstream scripts, notebooks, and models can load datasets without hardcoding file paths:

```python
from src.data import load_dataset

# Level 1: Probe space (p >> n)
df_train_probes = load_dataset("golub", level="probes", split="train")  # (38, 7131)
df_full_probes  = load_dataset("golub", level="probes", split="full")   # (72, 7131)

# Level 2: Collapsed gene space
df_train_genes  = load_dataset("golub", level="genes", split="train")   # (38, 5547)
df_arm_genes    = load_dataset("armstrong", level="genes")              # (103, 13239)

# Level 3: Cross-platform harmonized space (identical columns)
df_harm_golub     = load_dataset("golub", level="harmonized")           # (72, 5428)
df_harm_armstrong = load_dataset("armstrong", level="harmonized")       # (103, 5428)
```

### 5. Verification & Automated Testing

A dedicated test suite enforces invariant contracts and verifies data integrity across all pipeline stages:
1. **Dataset Dimensions & Integrity:** Validates expected matrix shapes across splits and cohorts.
2. **Leakage Prevention:** Asserts disjoint sample IDs across train/test splits, and validates `ProbeCollapser` zero-leakage behavior in scikit-learn pipelines.
3. **Class Balance Integrity:** Validates ground-truth distributions (*Train: 27 ALL / 11 AML; Test: 20 ALL / 14 AML*).
4. **Sanitization Bounds:** Asserts zero NaNs/Infs and verifies all probe intensities adhere to clipping bounds.
5. **Harmonization Invariants:** Verifies $\ge 5{,}000$ shared genes, 100% biomarker match, and column alignment.

#### Running Tests
```bash
pytest -v
```