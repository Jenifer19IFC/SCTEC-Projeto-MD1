import matplotlib.pyplot as plt
from src.config import FIGURES_DIR


def salvar_grafico(nome_arquivo: str) -> None:
    """Salva o gráfico atual na pasta outputs/figures"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    caminho = FIGURES_DIR / nome_arquivo

    plt.savefig(caminho, dpi=300, bbox_inches="tight")
    print(f"Gráfico salvo em: {caminho}")