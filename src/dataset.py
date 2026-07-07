import pandas as pd

from .config import DATASET_PATH


def carregar_dataset() -> pd.DataFrame:
    """Carrega o dataset bruto"""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset não encontrado em: {DATASET_PATH}")

    dataset = pd.read_csv(DATASET_PATH)
    print(f"Dataset carregado: {dataset.shape[0]} linhas x {dataset.shape[1]} colunas")
    return dataset
