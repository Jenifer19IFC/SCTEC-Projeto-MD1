"""Caminhos e constantes compartilhados pelo projeto"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR     = PROJECT_ROOT / "data" # Diretório de dados
RAW_DATA_DIR = DATA_DIR / "raw"      # Diretório de dados brutos

DATASET_FILENAME = "sleep_health_dataset.csv"      # Arq. do dataset bruto
DATASET_PATH     = RAW_DATA_DIR / DATASET_FILENAME # Diretorio completo do dataset bruto
