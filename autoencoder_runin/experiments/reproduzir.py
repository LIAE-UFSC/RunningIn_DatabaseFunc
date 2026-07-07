"""Script único de reprodução: gera todas as tabelas e figuras do estudo.

Produz, em ``outputs/experiments/``:
- Tabela 1 (CSV + markdown): {cru, PCA, autoencoder} × classificadores, multi-seed;
- Figura 2: UMAP do latente por unidade (colorido por tempo);
- Figura 3: ablação já existente (copiada de outputs/plots e outputs/heatmaps);
- Figura 4: probabilidade de amaciamento ao longo do tempo (grey zone) + sobreposição.

Reusa ``tabela`` e ``figuras``; não altera ``src/``. Rode a partir de
``autoencoder_runin/`` com o ambiente de ``requirements.txt``.
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
    """Copia os plots de ablação já existentes para a pasta do estudo (Figura 3)."""
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
    """Gera todas as tabelas e figuras do estudo em outputs/experiments/."""
    print("== Tabela 1 (multi-seed) ==")
    registros = rodar_multi_seed(seeds=seeds, hiperparametros=hiperparametros, verbose=True)
    agregado = agregar(registros)
    caminho_csv, caminho_md = exportar_tabela(agregado)
    print(f"  Tabela 1: {caminho_csv}\n            {caminho_md}")

    print("== Figura 2 (UMAP por unidade) ==")
    umap_path = gerar_umap_por_unidade(hiperparametros=hiperparametros)

    print("== Figura 3 (ablação existente) ==")
    ablacao = copiar_ablacao()
    print(f"  {len(ablacao)} plot(s) de ablação copiados para {EXPERIMENTS_OUTPUT_DIR / 'ablacao'}")

    print("== Figura 4 (grey zone) ==")
    greyzone_path, _ = gerar_figura_greyzone(hiperparametros=hiperparametros)
    overlay_path, instantes = gerar_figura_greyzone_overlay(hiperparametros=hiperparametros)

    print(f"\nConcluído. Assets em: {EXPERIMENTS_OUTPUT_DIR}")
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
