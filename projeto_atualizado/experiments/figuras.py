"""Geração das figuras do estudo a partir dos artefatos da validação cruzada.

Por enquanto: a figura UMAP por unidade (Figura 2), usando, para cada unidade, o
espaço latente projetado pelo autoencoder que **não a viu** durante o treino
(``latente_teste`` da dobra correspondente) — evidência visual da trajetória de
amaciamento e da generalização entre compressores.

Reusa ``runner`` e ``funcao_umap``; não altera ``src/``. Usa backend headless.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # geração de arquivo sem display

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.config import EXPERIMENTS_OUTPUT_DIR  # noqa: E402
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


if __name__ == "__main__":
    destino = gerar_umap_por_unidade()
    print(f"Figura salva em: {destino}")
