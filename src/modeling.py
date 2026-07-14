import warnings

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
import matplotlib.pyplot as plt
import pandas as pd
from .utils import salvar_grafico
import json
import joblib
from datetime import datetime
from .config import MODELS_DIR


def dividir_treino_teste(df, variavel_alvo, test_size=0.2, random_state=42):
    """
    Separa o df em treino (80%) e teste (20%)
    """

    X = df.drop(columns=variavel_alvo) # explicativas/preditoras (x)
    y = df[variavel_alvo]              # variável-alvo (y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print("=" * 60)
    print("DIVISÃO TREINO/TESTE")
    print("=" * 60)
    print(f"Tamanho do treino: {X_train.shape[0]}")
    print(f"Tamanho do teste : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


def treinar_regressao_linear(X, y):
    """
    Treina o modelo de Regressão Linear usando todos os dados disponíveis
    """
    colunas_categoricas = X.select_dtypes(include=["object", "category"]).columns
    colunas_numericas   = X.select_dtypes(exclude=["object", "category"]).columns

    pre_processador = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                colunas_categoricas,
            ),
            ("numericas", StandardScaler(), colunas_numericas),
        ]
    )

    modelo = Pipeline(
        steps=[
            ("pre_processador", pre_processador),
            ("regressao_linear", LinearRegression()),
        ]
    )

    # Treina o modelo com a base completa
    modelo.fit(X, y)

    return modelo

def realizar_predicao(modelo, X):
    """
    Realiza predições a partir do modelo treinado
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=".*matmul.*"
        )

        y_pred = modelo.predict(X)

    if not np.isfinite(y_pred).all():
        raise ValueError(
            "As predições possuem valores infinitos ou inválidos."
        )

    return y_pred


def avaliar_modelo(y_test, y_pred):
    """
    Calcula as métricas de avaliação do modelo
    """

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


def comparar_metricas_treino_teste(modelo, X_train, y_train, X_test, y_test):
    """
    Compara as métricas do modelo nos conjuntos de treino e teste
    """

    y_pred_train = modelo.predict(X_train)
    y_pred_test  = modelo.predict(X_test)

    metricas_treino = avaliar_modelo(y_train, y_pred_train)
    metricas_teste  = avaliar_modelo(y_test, y_pred_test)

    comparacao = pd.DataFrame(
        {
            "Treino": metricas_treino,
            "Teste": metricas_teste,
        }
    )

    comparacao["Diferença"] = comparacao["Teste"] - comparacao["Treino"]

    return comparacao


def plotar_valores_reais_previstos(y_test, y_pred):
    """
    Plota os valores reais x valores previstos
    """

    plt.figure(figsize=(6, 6))

    plt.scatter(y_test, y_pred, alpha=0.5)

    minimo = min(y_test.min(), y_pred.min())
    maximo = max(y_test.max(), y_pred.max())

    plt.plot([minimo, maximo], [minimo, maximo], color="red", linestyle="--")

    plt.xlabel("Valores reais")
    plt.ylabel("Valores previstos")
    plt.title("Valores reais x previstos")

    plt.tight_layout()
    salvar_grafico("valores_reais_previstos.png")
    plt.show()


def plotar_residuos(y_test, y_pred):
    """
    Plota a distribuição dos resíduos
    """

    residuos = y_test - y_pred

    plt.figure(figsize=(6, 4))

    plt.hist(residuos, bins=30, edgecolor="black")

    plt.axvline(0, color="red", linestyle="--")

    plt.xlabel("Resíduo")
    plt.ylabel("Frequência")
    plt.title("Distribuição dos resíduos")

    plt.tight_layout()
    salvar_grafico("distribuicao_residuos.png")
    plt.show()

def salvar_modelo(modelo, metricas, variaveis_explicativas):
    """
    Salva o modelo treinado e as métricas
    """

    versao       = obter_proxima_versao()
    pasta_versao = MODELS_DIR / versao

    pasta_versao.mkdir(parents=True, exist_ok=True)

    caminho_modelo   = (pasta_versao / f"modelo_regressao_{versao}.pkl")
    caminho_metricas = (pasta_versao / f"metricas_{versao}.json")

    joblib.dump(modelo, caminho_modelo)

    informacoes = {
        "versao": versao,
        "mae": float(metricas["MAE"]),
        "mse": float(metricas["MSE"]),
        "rmse": float(metricas["RMSE"]),
        "r2": float(metricas["R2"]),
        "data_treinamento": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "variaveis_explicativas": variaveis_explicativas
    }

    with open(caminho_metricas, "w", encoding="utf-8") as arquivo:
        json.dump(informacoes, arquivo, indent=4, ensure_ascii=False)

    print(f"Modelo salvo na versão {versao}.")
    print(f"Modelo: {caminho_modelo}")
    print(f"Métricas: {caminho_metricas}")


def obter_proxima_versao():
    """
    Verifica versões existentes e retorna a próxima versão
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    versoes = []

    for pasta in MODELS_DIR.iterdir():
        if pasta.is_dir() and pasta.name.startswith("v"):
            numero = pasta.name[1:]

            if numero.isdigit():
                versoes.append(int(numero))

    if not versoes:
        return "v1"

    return f"v{max(versoes) + 1}"
