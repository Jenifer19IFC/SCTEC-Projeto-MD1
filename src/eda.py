import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .utils import salvar_grafico

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


def visualizar_dados(df: pd.DataFrame) -> None:
    """Cria os gráficos analíticos"""
    alvo = "cognitive_performance_score"

    # 1. Histograma da variável-alvo (cognitive_performance_score)
    assimetria = df[alvo].skew()
    print(f"Assimetria da variável-alvo: {assimetria:.2f}")

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=alvo, bins=30, kde=True)
    plt.title(f"Distribuição da variável-alvo (assimetria = {assimetria:.2f})")
    plt.xlabel("Desempenho cognitivo")
    plt.ylabel("Frequência")
    salvar_grafico("distribuicao_variavel_alvo.png")
    plt.show()

    # 2. Dispersão: qualidade do sono e variável-alvo
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="sleep_quality_score",
        y=alvo,
        alpha=0.3,
    )
    plt.title("Qualidade do sono x Desempenho cognitivo")
    salvar_grafico("qualidade_sono_desempenho.png")
    plt.show()

    # 3. Dispersão: estresse e variável-alvo
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="stress_score",
        y=alvo,
        alpha=0.3,
    )
    plt.title("Pontuação de estresse x Desempenho cognitivo")
    salvar_grafico("estresse_desempenho.png")
    plt.show()

    # 4. Mapa de calor da correlação de Pearson
    correlacao = df.select_dtypes(include="number").corr(method="pearson")

    plt.figure(figsize=(16, 12))
    sns.heatmap(correlacao, cmap="coolwarm", center=0)
    plt.title("Correlação de Pearson entre as variáveis numéricas")
    salvar_grafico("correlacao_pearson.png")
    plt.show()

    print("Pares com correlação > 0.7:")
    pairs = correlacao.where(np.triu(np.ones(correlacao.shape), k=1).astype(bool)).stack()
    print(pairs[pairs.abs() > 0.7].sort_values(ascending=False))


def calcular_vif(df: pd.DataFrame, variavel_alvo: str = None) -> pd.DataFrame:
    """
    Calcula o Variance Inflation Factor (VIF) para as variáveis numéricas
    """

    # Seleciona apenas variáveis numéricas
    X = df.select_dtypes(include="number").copy()

    # Remove a variável-alvo
    if variavel_alvo is not None and variavel_alvo in X.columns:
        X = X.drop(columns=variavel_alvo)

    # Remove colunas constantes
    X = X.loc[:, X.nunique() > 1]

    vif = pd.DataFrame({
        "Variável": X.columns,
        "VIF": [
            variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])
        ]
    })

    return vif.sort_values("VIF", ascending=False).reset_index(drop=True)