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