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
from experiments.config import HIPERPARAMETROS, SEEDS, EXPERIMENTS_OUTPUT_DIR  # noqa: E402
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


def _formatar_celula(media, desvio):
    """Formata uma célula da tabela como ``média ± desvio`` (ou ``-`` se ausente)."""
    if pd.isna(media):
        return "-"
    if pd.isna(desvio):
        return f"{media:.3f}"
    return f"{media:.3f} ± {desvio:.3f}"


def montar_tabela(agregado, metricas=METRICAS_TABELA):
    """Pivota o agregado: linhas = condição×classificador, colunas = métricas.

    Cada célula é ``média ± desvio`` (entre as dobras).
    """
    df = agregado.copy()
    df["celula"] = [_formatar_celula(m, d) for m, d in zip(df["media"], df["desvio"])]
    pivot = df.pivot(index=["condicao", "classificador"], columns="metrica", values="celula")
    colunas = [m for m in metricas if m in pivot.columns]
    return pivot[colunas]


def _para_markdown(pivot):
    """Converte a tabela pivotada em texto de tabela markdown."""
    df = pivot.reset_index()
    headers = [str(c) for c in df.columns]
    linhas = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        linhas.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(linhas) + "\n"


def exportar_tabela(agregado, caminho_base=None, metricas=METRICAS_TABELA):
    """Exporta a Tabela 1 em CSV e markdown. Default: outputs/experiments/tabela1.*"""
    pivot = montar_tabela(agregado, metricas)
    base = Path(caminho_base) if caminho_base else EXPERIMENTS_OUTPUT_DIR / "tabela1"
    base.parent.mkdir(parents=True, exist_ok=True)

    caminho_csv = base.with_suffix(".csv")
    caminho_md = base.with_suffix(".md")
    pivot.to_csv(caminho_csv)
    caminho_md.write_text(_para_markdown(pivot), encoding="utf-8")
    return caminho_csv, caminho_md


if __name__ == "__main__":
    registros = rodar_multi_seed()
    agregado = agregar(registros)
    pd.set_option("display.width", 120)
    print("\nAgregado (média ± desvio entre as dobras):")
    print(agregado.to_string(index=False))

    caminho_csv, caminho_md = exportar_tabela(agregado)
    print(f"\nTabela 1 salva em:\n  {caminho_csv}\n  {caminho_md}")
