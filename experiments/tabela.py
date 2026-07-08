"""Multi-seed run of the three conditions and aggregation of the results.

Runs {cru, PCA, autoencoder} in the per-unit cross-validation, repeating over
several seeds to reduce estimation noise (mostly from the autoencoder), and
aggregates mean ± std **across folds** (generalization variability across
compressors).

Windowing is done once and reused across all runs.
Reuses ``runner`` and ``baselines``; does not modify ``src/``.
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
    """Run each condition on each seed and return a tidy DataFrame of records.

    Columns: condicao, seed, unit, classificador, metrica, valor.
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
                print(f"-> condition={nome_cond} | seed={seed}")
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
    """Aggregate mean ± std across folds (averaging over seeds per fold first).

    Returns:
        DataFrame with columns: condicao, classificador, metrica, media, desvio.
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
    """Format a table cell as ``mean ± std`` (or ``-`` if missing)."""
    if pd.isna(media):
        return "-"
    if pd.isna(desvio):
        return f"{media:.3f}"
    return f"{media:.3f} ± {desvio:.3f}"


def montar_tabela(agregado, metricas=METRICAS_TABELA):
    """Pivot the aggregate: rows = condition×classifier, columns = metrics.

    Each cell is ``mean ± std`` (across folds).
    """
    df = agregado.copy()
    df["celula"] = [_formatar_celula(m, d) for m, d in zip(df["media"], df["desvio"])]
    pivot = df.pivot(index=["condicao", "classificador"], columns="metrica", values="celula")
    colunas = [m for m in metricas if m in pivot.columns]
    return pivot[colunas]


def _para_markdown(pivot):
    """Convert the pivoted table into markdown table text."""
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
    """Export Table 1 as CSV and markdown. Default: outputs/experiments/tabela1.*"""
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
    print("\nAggregated (mean ± std across folds):")
    print(agregado.to_string(index=False))

    caminho_csv, caminho_md = exportar_tabela(agregado)
    print(f"\nTable 1 saved to:\n  {caminho_csv}\n  {caminho_md}")
