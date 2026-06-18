"""Execução multi-seed das três condições e agregação dos resultados.

Roda {cru, PCA, autoencoder} na validação cruzada por unidade, repetindo em
várias seeds para reduzir o ruído de estimativa (sobretudo do autoencoder), e
agrega média ± desvio **entre as dobras** (variabilidade de generalização entre
compressores).

O janelamento é feito uma única vez e reutilizado em todas as execuções.
Reusa ``runner`` e ``baselines``; não altera ``src/``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.config import HIPERPARAMETROS, SEEDS  # noqa: E402
from experiments.janelamento import janelar  # noqa: E402
from experiments.baselines import rodar_baseline_cru, rodar_baseline_pca  # noqa: E402
from experiments.runner import rodar_validacao_por_unidade  # noqa: E402

METRICAS_TABELA = ("F1_score", "Balanced_accuracy", "ROC_AUC", "PR_AUC", "MCC")

CONDICOES_PADRAO = {
    "cru": rodar_baseline_cru,
    "pca": rodar_baseline_pca,
    "autoencoder": rodar_validacao_por_unidade,
}


def rodar_multi_seed(seeds=None, hiperparametros=None, condicoes=None, verbose=True):
    """Roda cada condição em cada seed e devolve um DataFrame tidy de registros.

    Colunas: condicao, seed, unit, classificador, metrica, valor.
    """
    seeds = list(seeds) if seeds is not None else list(SEEDS)
    condicoes = condicoes if condicoes is not None else CONDICOES_PADRAO
    hp = {**HIPERPARAMETROS, **(hiperparametros or {})}

    df_janelado = janelar(
        n_amostras=hp["n_amostras"],
        janelamento=hp["janelamento"],
        amostras_repetidas=hp["amostras_repetidas"],
    )

    registros = []
    for nome_cond, funcao in condicoes.items():
        for seed in seeds:
            if verbose:
                print(f"-> condição={nome_cond} | seed={seed}")
            resultados = funcao(
                df_janelado=df_janelado,
                hiperparametros=hiperparametros,
                seed=seed,
                verbose=False,
            )
            for unit, por_clf in resultados.items():
                for clf, met in por_clf.items():
                    for metrica, valor in met.items():
                        if valor is None or isinstance(valor, (int, float)):
                            registros.append({
                                "condicao": nome_cond,
                                "seed": seed,
                                "unit": unit,
                                "classificador": clf,
                                "metrica": metrica,
                                "valor": np.nan if valor is None else float(valor),
                            })

    return pd.DataFrame(registros)


def agregar(df_registros, metricas=METRICAS_TABELA):
    """Agrega média ± desvio entre as dobras (média sobre seeds por dobra primeiro).

    Returns:
        DataFrame com colunas: condicao, classificador, metrica, media, desvio.
    """
    df = df_registros[df_registros["metrica"].isin(metricas)].copy()
    por_fold = (
        df.groupby(["condicao", "classificador", "metrica", "unit"])["valor"]
        .mean()
        .reset_index()
    )
    agregado = (
        por_fold.groupby(["condicao", "classificador", "metrica"])["valor"]
        .agg(media="mean", desvio="std")
        .reset_index()
    )
    return agregado


if __name__ == "__main__":
    registros = rodar_multi_seed()
    agregado = agregar(registros)
    pd.set_option("display.width", 120)
    print("\nAgregado (média ± desvio entre as dobras):")
    print(agregado.to_string(index=False))
