"""Generation of the study's figures from the cross-validation artifacts.

For now: the per-unit UMAP figure (Figure 2), using, for each unit, the latent space
projected by the autoencoder that **did not see it** during training (``latente_teste``
of the corresponding fold) — visual evidence of the run-in trajectory and of the
generalization across compressors.

Reuses ``runner`` and ``funcao_umap``; does not modify ``src/``. Uses a headless backend.
"""

import sys
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file generation without a display
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.config import EXPERIMENTS_OUTPUT_DIR, GREY_ZONE  # noqa: E402
from experiments.greyzone import probabilidade_por_unidade, estimar_instantes  # noqa: E402
from experiments.runner import rodar_validacao_por_unidade  # noqa: E402
from src.analysis.funcao_umap import plot_umap_latente_por_unidade  # noqa: E402


def gerar_umap_por_unidade(hiperparametros=None, seed=42, save_path=None,
                           min_dist=0.1, n_neighbors=15):
    """Generate the per-unit UMAP figure from the out-of-sample latents.

    Returns:
        Path of the saved figure.
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
        title="Latent space per unit — 2D UMAP (color = time; model did not see the unit)",
        save_path=caminho,
        min_dist=min_dist,
        n_neighbors=n_neighbors,
    )
    return caminho


def gerar_figura_greyzone(hiperparametros=None, seed=42, classificador="regressao_logistica",
                          suavizacao=15, save_path=None, n_cols=3):
    """Generate the figure of the run-in probability over time, per unit.

    Each subplot shows P(run-in) (raw and smoothed) on the unit's NA series, with the
    grey zone shaded. Uses the model that did not see the unit.

    Returns:
        (figure path, per-unit curves).
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
        ax.plot(curva["time"], curva["proba_suave"], color="#5b3fa8", linewidth=1.8, label="P smoothed")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Unit {unit}", fontsize=11)
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("P(run-in)", fontsize=10)
        ax.grid(True, color="#eeeeee", linewidth=0.6)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle("Run-in probability over time (model did not see the unit)", fontsize=13)
    fig.tight_layout()

    caminho = Path(save_path) if save_path else EXPERIMENTS_OUTPUT_DIR / "greyzone_probabilidade.png"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grey-zone figure saved to: {caminho}")
    return caminho, curvas


def gerar_figura_greyzone_overlay(hiperparametros=None, seed=42, classificador="regressao_logistica",
                                  suavizacao=15, limiar=0.5, save_path=None):
    """Overlay the units' smoothed P(run-in) and mark each one's estimated instant.

    Returns:
        (figure path, per-unit instants).
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
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("P(run-in) smoothed", fontsize=11)
    ax.set_title("Run-in transition per unit (dashed = estimated instant)", fontsize=13)
    ax.grid(True, color="#eeeeee", linewidth=0.6)
    ax.legend(fontsize=9, ncol=2, loc="lower right")
    fig.tight_layout()

    caminho = Path(save_path) if save_path else EXPERIMENTS_OUTPUT_DIR / "greyzone_overlay.png"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Overlay figure saved to: {caminho}")
    return caminho, instantes


if __name__ == "__main__":
    destino = gerar_umap_por_unidade()
    print(f"UMAP figure saved to: {destino}")
    destino_gz, _ = gerar_figura_greyzone()
    print(f"Grey-zone figure saved to: {destino_gz}")
    destino_ov, instantes = gerar_figura_greyzone_overlay()
    print(f"Overlay figure saved to: {destino_ov}")
    print(f"Estimated instants: {instantes}")
