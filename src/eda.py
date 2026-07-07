import pandas as pd


def estatistica_descritiva(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DIMENSÕES DO DATASET")
    print("=" * 60)
    print(f"Linhas : {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")

    print("\n")

    print("=" * 60)
    print("TIPOS DAS VARIÁVEIS")
    print("=" * 60)
    print(df.dtypes)

    print("\n")

    print("=" * 60)
    print("ESTATÍSTICAS DESCRITIVAS")
    print("=" * 60)
    display(df.describe())