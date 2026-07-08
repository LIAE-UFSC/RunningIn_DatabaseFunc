"""Single reproduction script: generates all the study's tables and figures.

Produces, in ``outputs/experiments/``:
- Table 1 (CSV + markdown): {cru, PCA, autoencoder} × classifiers, multi-seed;
- Figure 2: per-unit latent UMAP (colored by time);
- Figure 3: existing ablation (copied from outputs/plots and outputs/heatmaps);
- Figure 4: run-in probability over time (grey zone) + overlay.

Reuses ``tabela`` and ``figuras``; does not modify ``src/``. Run from the repository
root with the environment from ``requirements.txt``.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import PLOTS_DIR, HEATMAPS_DIR  # noqa: E402
from experiments.config import EXPERIMENTS_OUTPUT_DIR  # noqa: E402
from experiments.tabela import rodar_multi_seed, agregar, exportar_tabela  # noqa: E402
from experiments.figuras import (  # noqa: E402
    gerar_umap_por_unidade,
    gerar_figura_greyzone,
    gerar_figura_greyzone_overlay,
)


def copiar_ablacao(destino=None):
    """Copy the existing ablation plots into the study folder (Figure 3)."""
    destino = Path(destino) if destino else EXPERIMENTS_OUTPUT_DIR / "ablacao"
    destino.mkdir(parents=True, exist_ok=True)

    origens = list(PLOTS_DIR.glob("*_separado.png")) + list(HEATMAPS_DIR.glob("*.png"))
    copiados = []
    for origem in origens:
        alvo = destino / origem.name
        shutil.copy2(origem, alvo)
        copiados.append(alvo)
    return copiados


def reproduzir_tudo(hiperparametros=None, seeds=None):
    """Generate all the study's tables and figures in outputs/experiments/."""
    print("== Table 1 (multi-seed) ==")
    registros = rodar_multi_seed(seeds=seeds, hiperparametros=hiperparametros, verbose=True)
    agregado = agregar(registros)
    caminho_csv, caminho_md = exportar_tabela(agregado)
    print(f"  Table 1: {caminho_csv}\n           {caminho_md}")

    print("== Figure 2 (per-unit UMAP) ==")
    umap_path = gerar_umap_por_unidade(hiperparametros=hiperparametros)

    print("== Figure 3 (existing ablation) ==")
    ablacao = copiar_ablacao()
    print(f"  {len(ablacao)} ablation plot(s) copied to {EXPERIMENTS_OUTPUT_DIR / 'ablacao'}")

    print("== Figure 4 (grey zone) ==")
    greyzone_path, _ = gerar_figura_greyzone(hiperparametros=hiperparametros)
    overlay_path, instantes = gerar_figura_greyzone_overlay(hiperparametros=hiperparametros)

    print(f"\nDone. Assets in: {EXPERIMENTS_OUTPUT_DIR}")
    return {
        "tabela_csv": caminho_csv,
        "tabela_md": caminho_md,
        "umap": umap_path,
        "ablacao": ablacao,
        "greyzone": greyzone_path,
        "greyzone_overlay": overlay_path,
        "instantes": instantes,
    }


if __name__ == "__main__":
    reproduzir_tudo()
