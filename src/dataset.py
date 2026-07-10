import pandas as pd

from .config import DATASET_PATH, FINAL_DATA_DIR, PROCESSED_DATA_DIR


def carregar_dataset() -> pd.DataFrame:
    """Carrega o dataset bruto"""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset não encontrado em: {DATASET_PATH}")

    # Mostra todas as colunas ao exibir o df
    pd.set_option("display.max_columns", None)

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df


def salvar_dataset_processado(df: pd.DataFrame, versao: str = "v1") -> None:
    """Salva o dataset limpo/tratado"""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    caminho = PROCESSED_DATA_DIR / f"dataset_processado_{versao}.csv"
    df.to_csv(caminho, index=False)

    print(f"Dataset processado salvo em: {caminho}")


def salvar_dataset_final(df: pd.DataFrame, versao: str = "v1") -> None:
    """Salva o recorte usado na modelagem"""

    FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    caminho = FINAL_DATA_DIR / f"dataset_final_{versao}.csv"
    df.to_csv(caminho, index=False)

    print(f"Dataset final salvo em: {caminho}")
