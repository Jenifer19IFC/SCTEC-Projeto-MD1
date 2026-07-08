import pandas as pd

from .config import DATASET_PATH


def carregar_dataset() -> pd.DataFrame:
    """Carrega o dataset bruto"""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset não encontrado em: {DATASET_PATH}")

    # Mostra todas as colunas ao exibir o df
    pd.set_option("display.max_columns", None)

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df
