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

---

## Quickstart Workflow

Run the complete end-to-end data acquisition, preprocessing, harmonization, and EDA visualization pipeline:

```bash
# 1. Download raw cohorts (Golub via Kaggle Hub, Coustan-Smith via NCBI GEO)
python -m src.data.download_data --dataset golub
python -m src.data.download_data --dataset coustan_smith

# 2. Ingest, transpose, clip [20, 16000], and impute expression matrices
python -m src.data.make_dataset

# 3. Map probes to HUGO symbols, collapse duplicates, and harmonize cohorts
python -m src.data.map_genes --collapse --harmonize

# 4. Generate IEEE conference-ready figures (Grayscale or Academic palette)
python -m src.visualization.eda --style grayscale

# 5. Run automated test suite (45 unit and invariant verification tests)
pytest -v
```

---

## Repository Structure

```text
leukemia-subtype-benchmark/
├── data/
│   ├── external/               # Platform annotation files (e.g. GPL80)
│   ├── processed/              # Parquet matrices by level (probes, genes, harmonized)
│   └── raw/                    # Downloaded raw archives (golub, coustan_smith_2011)
├── docs/
│   ├── appendix_b_technical_notes.md  # Mathematical formulations & CV architecture
│   ├── genai_log.md                   # Appendix A: GenAI traceability & audit log
│   └── references_verified.md         # 15 verified references with persistent DOIs
├── outputs/
│   ├── figures/                # Publication-grade figures (Vector PDF & 300 DPI PNG)
│   └── tables/                 # Harmonization and biomarker mapping CSV/Markdown
├── scripts/
│   └── setup_kaggle_auth.py    # Interactive Kaggle credentials configurator
├── src/
│   ├── config.py               # Global seeds, paths, and platform metadata
│   ├── data/                   # Acquisition, preprocessing, and gene mapping modules
│   ├── features/               # Zero-leakage scikit-learn transformers (ProbeCollapser)
│   ├── models/                 # Model evaluation and nested CV engines (Phase II)
│   └── visualization/          # Publication-grade EDA plotting and styling engine
├── tests/                      # Automated unit and invariant test suite (45 tests)
├── environment.yml             # Conda environment specification
└── requirements.txt            # Pinned pip dependencies
```

---

## Data Pipeline & Preprocessing

The benchmark evaluates models on leukemia microarray cohorts framed strictly as a binary subtype classification problem (**ALL vs. AML**):
- **Golub et al. (1999):** Primary benchmark ($n=72$, $p=7{,}129$). Staged via Kaggle Hub.
- **Coustan-Smith et al. (2011) / GSE28497:** External validation cohort from NCBI GEO.
- **MILE Study / GSE13159:** Large-scale cross-study cohort from NCBI GEO.

### 1. Automated Data Acquisition

Datasets can be acquired programmatically using the download pipeline without manually downloading or placing files:

```bash
# Download all cohorts (Golub, Coustan-Smith, MILE)
python -m src.data.download_data

# Or download a specific dataset:
python -m src.data.download_data --dataset golub
python -m src.data.download_data --dataset coustan_smith
python -m src.data.download_data --dataset mile
```

> [!NOTE]
> **Kaggle API Token (Golub Dataset):**
> Acquiring the Golub dataset requires a Kaggle API token. You can configure it interactively by running:
> ```bash
> python scripts/setup_kaggle_auth.py
> ```
> Alternatively, set `KAGGLE_API_TOKEN` in a `.env` file at the project root or in your shell environment. If not detected when running the download script, you will be prompted to configure it.
> GEO datasets (*Coustan-Smith* and *MILE*) are retrieved publicly via NCBI GEO without authentication.

Downloaded raw archives and tables are automatically organized into dedicated directories under `data/raw/`:
- `data/raw/golub/`
- `data/raw/coustan_smith_2011/`
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
python -m src.data.make_dataset --dataset coustan_smith
```

### 3. Probe-to-Gene Mapping & Cross-Platform Harmonization

Because different microarray platforms use distinct probe identifier systems, cross-cohort validation requires mapping probes to standardized **HUGO gene symbols** and harmonizing overlapping feature spaces:
- **Golub et al. (1999):** Affymetrix HuGeneFL / Hu6800 (GEO GPL80) $\to$ 7,129 probes mapped to 5,545 unique HUGO symbols.
- **Coustan-Smith et al. (2011) / GSE28497:** Affymetrix Human Genome U133A (GEO GPL96) $\to$ 22,283 probes mapped to 13,237 unique HUGO symbols.
- **Biomarker Validation:** 100% verified cross-platform match for key leukemia biomarkers (*CST3*, *CD33*, *CD19*, *MPO*, *FAH*, *MME/CD10*, *LYZ*).
- **Cross-Platform Intersection:** **5,426 shared HUGO genes** (97.9% coverage of Golub's mapped genes).

#### Zero-Leakage Probe Collapsing
When multiple probes target the same gene, `ProbeCollapser` (`sklearn.base.BaseEstimator`, `TransformerMixin`) aggregates them into a single gene feature. To prevent data leakage:
- When using `max_variance`, probe variances are computed **strictly on training fold samples** inside the pipeline.
- On benchmark splits, probe selection fits strictly on `golub_train_38` and transforms `golub_test_34`.
- Row-wise `mean` and `median` strategies are also supported.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.features.collapser import ProbeCollapser

# Seamless integration into scikit-learn cross-validation pipelines:
pipe = Pipeline([
    ("collapser", ProbeCollapser(
        probe_to_gene_path="data/processed/annotations/golub_probe_to_gene.parquet",
        strategy="max_variance"
    )),
    ("classifier", LogisticRegression(penalty="l1", solver="saga", max_iter=2000))
])

# Probe selection is fit strictly on training fold data:
pipe.fit(X_train_probes, y_train)
y_pred = pipe.predict(X_test_probes)
```

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
│   └── coustan_smith_full.parquet
│
├── genes/                # Level 2: Collapsed HUGO gene-level matrices
│   ├── golub_train_genes.parquet
│   ├── golub_test_genes.parquet
│   ├── golub_full_genes.parquet
│   └── coustan_smith_full_genes.parquet
│
├── harmonized/           # Level 3: Cross-platform harmonized matrices (5426 shared genes)
│   ├── golub_harmonized_genes.parquet
│   └── coustan_smith_harmonized_genes.parquet
│
└── annotations/          # Platform probe-to-gene lookup tables (.parquet & .csv)
    ├── golub_probe_to_gene.parquet
    └── coustan_smith_probe_to_gene.parquet
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
df_cs_genes    = load_dataset("coustan_smith", level="genes")              # (103, 13239)

# Level 3: Cross-platform harmonized space (identical columns)
df_harm_golub     = load_dataset("golub", level="harmonized")           # (72, 5428)
df_harm_coustan = load_dataset("coustan_smith", level="harmonized")       # (103, 5428)
```

### 5. Verification & Automated Testing

A dedicated test suite of 45 automated unit tests enforces invariant contracts and verifies data and visualization integrity across all pipeline stages:
1. **Dataset Dimensions & Integrity:** Validates expected matrix shapes across splits and cohorts.
2. **Leakage Prevention:** Asserts disjoint sample IDs across train/test splits, and validates `ProbeCollapser` zero-leakage behavior in scikit-learn pipelines.
3. **Class Balance Integrity:** Validates ground-truth distributions (*Train: 27 ALL / 11 AML; Test: 20 ALL / 14 AML*).
4. **Sanitization Bounds:** Asserts zero NaNs/Infs and verifies all probe intensities adhere to clipping bounds.
5. **Harmonization Invariants:** Verifies $\ge 5{,}000$ shared genes, 100% biomarker match, and column alignment.
6. **EDA Engine Assertions:** Verifies generation of publication-grade figures, statistical skewness bounds, PCA variance consistency, and biomarker recovery.

#### Running Tests
```bash
pytest -v
```

---

## Exploratory Data Analysis (EDA) & Visualization Engine

The benchmark features a modular visualization engine ([`src/visualization/eda.py`](file:///c:/Users/Ozbay/Desktop/leukemia-subtype-benchmark/src/visualization/eda.py)) and styling framework ([`src/visualization/style.py`](file:///c:/Users/Ozbay/Desktop/leukemia-subtype-benchmark/src/visualization/style.py)) engineered for publication-grade scientific graphics. All figures are dual-exported as 300+ DPI raster PNGs and scalable vector PDFs to `outputs/figures/`.

### 1. Style Presets & Theming

Figures support three cohesive styling presets designed for different publishing contexts:

* **`grayscale` (Default):** Strictly monochromatic palette with custom marker geometries (`o`, `s`, `^`), line styles (`-`, `--`, `:`), and hatching patterns (`//`, `xx`) optimized for high-contrast, black-and-white print and IEEE conference compliance.
* **`academic`:** Subdued, colorblind-friendly scientific palette (`coolwarm`, `crest`, `tab10`) tailored for multi-panel journal articles and digital reading.
* **`vibrant`:** High-contrast saturated palette optimized for slide decks, conference presentations, and interactive dashboards.

### 2. Command-Line Execution

Generate the complete figure suite with a single command:

```bash
# Generate all figures using default monochrome/grayscale IEEE style
python -m src.visualization.eda --style grayscale

# Generate using academic or vibrant color palettes
python -m src.visualization.eda --style academic
python -m src.visualization.eda --style vibrant

# Target a specific cohort (golub or coustan_smith)
python -m src.visualization.eda --dataset golub --style academic
```

### 3. Generated Visual Artifacts & Scientific Takeaways

| Artifact | Script / Function | Methods & Parameters | Key Scientific Finding |
|:---|:---|:---|:---|
| **Figure 1**<br>`fig1_expression_distributions` | `plot_expression_distributions()` | 4-panel KDE distributions pre- and post-$\log_2(x+1)$ transformation on clipped fluorescence. | Raw Affymetrix arrays exhibit severe right-skewness ($6.03$ Golub, $6.06$ Armstrong). $\log_2(x+1)$ transformation stabilizes skewness to $0.57$ and $0.25$, yielding symmetric Gaussian-like profiles and stabilizing inter-gene variance. |
| **Figure 2**<br>`fig2_dimensionality_reduction` | `plot_dimensionality_reduction()` | Scree plot (top 15 PCs), 2D PCA, 2D t-SNE ($\text{perplexity}=15$), 2D UMAP ($\text{metric}=\text{'cosine'}$). | Top 2 linear PCs capture only $20.5\%$ variance (PC1: $11.9\%$, PC2: $8.6\%$) with partial ALL/AML overlap due to high-dimensional background noise. Non-linear manifolds (t-SNE and UMAP) achieve clean unsupervised cluster separation without label supervision. |
| **Figure 3**<br>`fig3_top_genes_heatmap` | `plot_top_genes_heatmap()` | Two-sample Welch's $t$-test + Mann-Whitney $U$ test, Benjamini-Hochberg FDR ($\alpha=0.05$). Two-way hierarchical clustering (Euclidean, Ward linkage) on top 40 genes. | Clustermap dendrogram partitions all $n=72$ samples into distinct ALL and AML branches with high purity. Data-driven selection blindly recovers canonical AML markers (*LYZ*, *CST3*, *MPO*, *FAH*, *CD33*) and ALL markers (*CD19*). |
| **Figure 4**<br>`fig4_batch_effect_diagnostics` | `plot_batch_effect_diagnostics()` | Panel (a): Golub Train ($n=38$) vs Test ($n=34$) joint PCA.<br>Panel (b): Cross-platform Golub (Hu6800, $n=72$) vs Coustan-Smith (HG-U133A, $n=103$) PCA on 5,426 shared genes. | Exposes sample shift and variance discrepancy between fixed 38/34 splits, proving single-split benchmark unreliability. Confirms within-cohort z-score standardization successfully centers multi-generation Affymetrix platforms into a shared space. |

### 4. Programmatic Python API

Each visualization function is modular, decoupled from file I/O, and callable directly in custom scripts or Jupyter notebooks:

```python
import matplotlib.pyplot as plt
from src.data import load_dataset
from src.visualization.style import apply_style, save_figure
from src.visualization.eda import (
    plot_expression_distributions,
    plot_dimensionality_reduction,
    plot_top_genes_heatmap,
    plot_batch_effect_diagnostics,
)

# 1. Activate styling preset
apply_style("academic")

# 2. Load data via accessor API
df_golub = load_dataset("golub", level="probes", split="full")
df_coustan = load_dataset("coustan_smith", level="probes")

# 3. Generate figures programmatically
fig1 = plot_expression_distributions(df_golub, df_coustan, style="academic")
fig2 = plot_dimensionality_reduction(df_golub, style="academic")
fig3 = plot_top_genes_heatmap(df_golub, top_n=40, style="academic")

# 4. Save dual vector PDF + 300 DPI raster PNG
save_figure(fig1, "fig1_custom_export", output_dir="outputs/figures", preset="academic")
plt.show()
```

---

## Documentation & Technical Appendices

Detailed methodological formulations, bibliographies, and compliance audits are maintained under `docs/`:

* **`docs/references_verified.md`**: Systematic literature review bibliography containing complete, verified publication metadata and persistent DOI hyperlinks for all 15 benchmarked studies.
* **`docs/genai_log.md`**: **Appendix A** (GenAI Disclosure & Traceability Log) detailing prompt sequences, verification protocols, and human oversight declarations.
* **`docs/appendix_b_technical_notes.md`**: **Appendix B** (Technical Notes & Mathematical Formulations) detailing the mathematics of probe collapsing, information-theoretic Mutual Information feature selection, and the repeated $5 \times 5$ nested cross-validation protocol.
