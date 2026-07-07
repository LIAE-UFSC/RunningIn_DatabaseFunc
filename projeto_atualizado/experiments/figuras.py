"""Geração das figuras do estudo a partir dos artefatos da validação cruzada.

Por enquanto: a figura UMAP por unidade (Figura 2), usando, para cada unidade, o
espaço latente projetado pelo autoencoder que **não a viu** durante o treino
(``latente_teste`` da dobra correspondente) — evidência visual da trajetória de
amaciamento e da generalização entre compressores.

Reusa ``runner`` e ``funcao_umap``; não altera ``src/``. Usa backend headless.
"""

import sys
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # geração de arquivo sem display
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.config import EXPERIMENTS_OUTPUT_DIR, GREY_ZONE  # noqa: E402
from experiments.greyzone import probabilidade_por_unidade, estimar_instantes  # noqa: E402
from experiments.runner import rodar_validacao_por_unidade  # noqa: E402
from src.analysis.funcao_umap import plot_umap_latente_por_unidade  # noqa: E402


def gerar_umap_por_unidade(hiperparametros=None, seed=42, save_path=None,
                           min_dist=0.1, n_neighbors=15):
    """Gera a figura UMAP por unidade a partir dos latentes fora-da-amostra.

    Returns:
        Caminho da figura salva.
    """
    _, artefatos = rodar_validacao_por_unidade(
        hiperparametros=hiperparametros,
        seed=seed,
        verbose=False,
        retornar_artefatos=True,
    )
    latentes = {unit: art["latente_teste"] for unit, art in artefatos.items()}

    caminho = Path(save_path) if save_path else EXPERIMENTS_OUTPUT_DIR / "umap_por_unidade.png"
    caminho.parent.mkdir(parents=True, exist_ok=True)

    plot_umap_latente_por_unidade(
        latentes,
        title="Espaço latente por unidade — UMAP 2D (cor = tempo; modelo não viu a unidade)",
        save_path=caminho,
        min_dist=min_dist,
        n_neighbors=n_neighbors,
    )
    return caminho


def gerar_figura_greyzone(hiperparametros=None, seed=42, classificador="regressao_logistica",
                          suavizacao=15, save_path=None, n_cols=3):
    """Gera a figura da probabilidade de amaciamento ao longo do tempo, por unidade.

    Cada subplot mostra P(amaciado) (bruto e suavizado) na série NA da unidade, com a
    grey zone sombreada. Usa o modelo que não viu a unidade.

    Returns:
        (caminho da figura, curvas por unidade).
    """
    curvas = probabilidade_por_unidade(
        hiperparametros=hiperparametros, seed=seed,
        classificador=classificador, suavizacao=suavizacao,
    )

    unidades = list(curvas.keys())
    n = len(unidades)
    n_cols = min(n_cols, n)
    n_rows = ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.6 * n_rows), squeeze=False)

    for idx, unit in enumerate(unidades):
        ax = axes[idx // n_cols][idx % n_cols]
        curva = curvas[unit]
        ax.axvspan(GREY_ZONE[0], GREY_ZONE[1], color="#bbbbbb", alpha=0.35, label="grey zone")
        ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle=":")
        ax.plot(curva["time"], curva["proba"], color="#c9c2e8", linewidth=0.8, alpha=0.7)
        ax.plot(curva["time"], curva["proba_suave"], color="#5b3fa8", linewidth=1.8, label="P suavizado")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Unidade {unit}", fontsize=11)
        ax.set_xlabel("Tempo", fontsize=10)
        ax.set_ylabel("P(amaciado)", fontsize=10)
        ax.grid(True, color="#eeeeee", linewidth=0.6)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle("Probabilidade de amaciamento ao longo do tempo (modelo não viu a unidade)", fontsize=13)
    fig.tight_layout()

    caminho = Path(save_path) if save_path else EXPERIMENTS_OUTPUT_DIR / "greyzone_probabilidade.png"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura da grey zone salva em: {caminho}")
    return caminho, curvas


def gerar_figura_greyzone_overlay(hiperparametros=None, seed=42, classificador="regressao_logistica",
                                  suavizacao=15, limiar=0.5, save_path=None):
    """Sobrepõe P(amaciado) suavizado das unidades e marca o instante estimado de cada uma.

    Returns:
        (caminho da figura, instantes por unidade).
    """
    curvas = probabilidade_por_unidade(
        hiperparametros=hiperparametros, seed=seed,
        classificador=classificador, suavizacao=suavizacao,
    )
    instantes = estimar_instantes(curvas, limiar=limiar)

    cores = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axvspan(GREY_ZONE[0], GREY_ZONE[1], color="#bbbbbb", alpha=0.3, label="grey zone")
    ax.axhline(limiar, color="#888888", linewidth=0.8, linestyle=":")

    for i, (unit, curva) in enumerate(curvas.items()):
        cor = cores[i % len(cores)]
        ax.plot(curva["time"], curva["proba_suave"], color=cor, linewidth=1.8, label=unit)
        inst = instantes[unit]
        if inst is not None:
            ax.axvline(inst, color=cor, linewidth=1.0, linestyle="--", alpha=0.7)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Tempo", fontsize=11)
    ax.set_ylabel("P(amaciado) suavizado", fontsize=11)
    ax.set_title("Transição de amaciamento por unidade (traço = instante estimado)", fontsize=13)
    ax.grid(True, color="#eeeeee", linewidth=0.6)
    ax.legend(fontsize=9, ncol=2, loc="lower right")
    fig.tight_layout()

    caminho = Path(save_path) if save_path else EXPERIMENTS_OUTPUT_DIR / "greyzone_overlay.png"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura sobreposta salva em: {caminho}")
    return caminho, instantes


if __name__ == "__main__":
    destino = gerar_umap_por_unidade()
    print(f"Figura UMAP salva em: {destino}")
    destino_gz, _ = gerar_figura_greyzone()
    print(f"Figura grey zone salva em: {destino_gz}")
    destino_ov, instantes = gerar_figura_greyzone_overlay()
    print(f"Figura sobreposta salva em: {destino_ov}")
    print(f"Instantes estimados: {instantes}")
