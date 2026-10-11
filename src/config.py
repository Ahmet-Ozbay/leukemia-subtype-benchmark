"""
Central configuration for Leukemia Subtype Benchmark.
Defines project paths, reproducibility seeds, and label conventions.
"""

import os
from pathlib import Path

# ==============================================================================
# Directory Structure
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Cohort-specific raw directories
GOLUB_RAW_DIR = RAW_DATA_DIR / "golub"
ARMSTRONG_RAW_DIR = RAW_DATA_DIR / "armstrong_2002"
MILE_RAW_DIR = RAW_DATA_DIR / "mile_study"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Ensure runtime directories exist
for path in [
    RAW_DATA_DIR,
    GOLUB_RAW_DIR,
    ARMSTRONG_RAW_DIR,
    MILE_RAW_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Cross-Platform Environment Variable Loader (.env)
# ==============================================================================
def load_dotenv(env_path: Path = PROJECT_ROOT / ".env") -> None:
    """Loads key-value pairs from .env into os.environ without third-party dependencies."""
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key not in os.environ:
                os.environ[key] = val


load_dotenv()

# ==============================================================================
# Reproducibility & Seeds (Mandated by Section 4 & 6.9)
# ==============================================================================
GLOBAL_SEED = 42
CV_SEEDS = [42, 123, 456, 789, 101112]

# ==============================================================================
# Domain & Class Definitions
# ==============================================================================
LABEL_MAP = {
    "ALL": 0,
    "AML": 1,
}

INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

EXPRESSION_CLIP_MIN = 20.0
EXPRESSION_CLIP_MAX = 16000.0