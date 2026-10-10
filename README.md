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

The primary benchmark is based on the **Golub et al. (1999)** microarray dataset, framed strictly as a binary leukemia subtype classification problem (**ALL vs. AML**).

### Preprocessing Protocol
1. **Matrix Transposition:** Converts raw gene-by-sample matrices into standard sample-by-feature layout ($n=72$, $p=7{,}129$).
2. **Metadata Sanitization:** Drops non-numeric call confidence flags (`P`, `A`, `M`) and aligns patient identifiers against ground truth labels without relying on column ordering.
3. **Thresholding & Imputation:** Clips expression intensities to $[20.0, 16000.0]$ to suppress background noise and sensor saturation, followed by median imputation for any remaining missing values.
4. **Serialization:** Exports cleaned partitions to `data/processed/` as high-performance Parquet files.

### Execution

To run the data ingestion and transformation pipeline:

```bash
python -m src.data.make_dataset
```
### Verification & Automated Testing

A dedicated test suite enforces invariant contracts and verifies data integrity against the assignment specifications:
1. **Matrix Transposition:** Converts raw gene-by-sample matrices into standard sample-by-feature layout ($n=72$, $p=7{,}129$).
2. **Leakage Prevention:** Asserts patient IDs across train and test partitions are completely disjoint.
3. **Class Balance Integrity:** Validates ground-truth distributions (*Train: 27 ALL / 11 AML; Test: 20 ALL / 14 AML*).
4. **Sanitization Bounds:** Asserts zero NaNs/Infs and verifies all probe intensities adhere to clipping bounds.

#### Running Tests
```bash
pytest -v
```