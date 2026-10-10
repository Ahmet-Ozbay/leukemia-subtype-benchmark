"""
Central configuration for Leukemia Subtype Benchmark.
Defines project paths, reproducibility seeds, and label conventions.
"""

from pathlib import Path

# ==============================================================================
# Directory Structure
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Ensure runtime directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, FIGURES_DIR, TABLES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Reproducibility & Seeds (Mandated by Section 4 & 6.9)
# ==============================================================================
GLOBAL_SEED = 42
CV_SEEDS = [42, 123, 456, 789, 101112]

# ==============================================================================
# Domain & Class Definitions
# ==============================================================================
# ALL = Acute Lymphoblastic Leukemia (0)
# AML = Acute Myeloid Leukemia (1)
LABEL_MAP = {
    "ALL": 0,
    "AML": 1
}

INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Golub standard preprocessing thresholds
EXPRESSION_CLIP_MIN = 20.0
EXPRESSION_CLIP_MAX = 16000.0