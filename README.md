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
python -c "import sklearn, shap, umap, optuna, GEOparse; print('Environment successfully initialized!')"
```