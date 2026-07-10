"""Caminhos e constantes compartilhados pelo projeto"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dados
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"

# Saídas
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Modelos
MODELS_DIR = PROJECT_ROOT / "models"

# Dataset
DATASET_FILENAME = "sleep_health_dataset.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME
