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
4. **Serialization:** Exports cleaned partitions to `data/processed/` as high-performance Parquet files.

#### Running Preprocessing

To execute data transformation and preprocessing:

```bash
# Process all downloaded cohorts
python -m src.data.make_dataset

# Or process a specific cohort:
python -m src.data.make_dataset --dataset golub
```

### 3. Verification & Automated Testing

A dedicated test suite enforces invariant contracts and verifies data integrity against benchmark specifications:
1. **Dataset Dimensions:** Validates expected matrix shapes across splits ($38 \times 7{,}131$ train, $34 \times 7{,}131$ test, $72 \times 7{,}131$ full).
2. **Leakage Prevention:** Asserts patient IDs across train and test partitions are completely disjoint.
3. **Class Balance Integrity:** Validates ground-truth distributions (*Train: 27 ALL / 11 AML; Test: 20 ALL / 14 AML*).
4. **Sanitization Bounds:** Asserts zero NaNs/Infs and verifies all probe intensities adhere to clipping bounds.

#### Running Tests
```bash
pytest -v
```