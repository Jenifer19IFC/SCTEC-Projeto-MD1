import pandas as pd

def verificar_duplicados(df: pd.DataFrame) -> int:
    """
    Verifica a presença de registros duplicados em um df
    """
    duplicados = df.duplicated().sum()

    print("=" * 60)
    print("VERIFICAÇÃO DE DUPLICADOS")
    print("=" * 60)
    print(f"Registros duplicados: {duplicados}")

    return duplicados


import pandas as pd


def verificar_valores_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica a quantidade e o percentual de valores ausentes por coluna
    """
    ausentes = pd.DataFrame({
        "Valores Ausentes": df.isna().sum(),
        "Percentual (%)": (df.isna().mean() * 100).round(2)
    })

    ausentes = ausentes[ausentes["Valores Ausentes"] > 0]

    print("=" * 60)
    print("VERIFICAÇÃO DE VALORES AUSENTES")
    print("=" * 60)

    if ausentes.empty:
        print("Não há valores ausentes no conjunto de dados.")
    else:
        print(ausentes)

    return ausentes