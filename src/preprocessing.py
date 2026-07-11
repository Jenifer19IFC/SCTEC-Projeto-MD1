import math
import matplotlib.pyplot as plt
import pandas as pd
from .utils import salvar_grafico

def verificar_duplicados(df: pd.DataFrame) -> int:
    """
    Verifica a presença de registros duplicados no df
    """
    duplicados = df.duplicated().sum()

    print("=" * 60)
    print("VERIFICAÇÃO DE DUPLICADOS")
    print("=" * 60)
    print(f"Registros duplicados: {duplicados}")

    return duplicados

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

import pandas as pd


def remover_colunas(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """
    Remove uma ou + colunas do df
    """
    colunas_existentes = [coluna for coluna in colunas if coluna in df.columns]

    if colunas_existentes:
        print(f"Colunas removidas: {', '.join(colunas_existentes)}")
    else:
        print("Nenhuma das colunas informadas foi encontrada.")

    return df.drop(columns=colunas_existentes)


def plotar_boxplots(df: pd.DataFrame, variavel_alvo: str = None, nome_arquivo: str = "boxplots_variaveis_explicativas.png") -> None:
    """
    Plota boxplots das variáveis numéricas explicativas em um painel
    """

    # Seleciona somente variáveis numéricas
    colunas = df.select_dtypes(include="number").columns.tolist()

    # Remove a variável-alvo da lista de colunas para não plotar no boxplot
    if variavel_alvo in colunas: 
        colunas.remove(variavel_alvo)

    # Remove variáveis binárias 
    colunas = [coluna for coluna in colunas if df[coluna].nunique() > 2]

    n_cols = 4
    n_rows = math.ceil(len(colunas) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    fig.suptitle("Boxplots das Variáveis Numéricas Explicativas", fontsize=18, fontweight="bold")

    axes = axes.flatten()

    for ax, coluna in zip(axes, colunas):
        ax.boxplot(df[coluna].dropna(), vert=False)
        ax.set_title(coluna, fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Remove eixos vazios
    for ax in axes[len(colunas):]:
        fig.delaxes(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    salvar_grafico(nome_arquivo)

    plt.show()