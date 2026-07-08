"""Verification that windowing and splits are leakage-free.

Checks the invariants that guarantee an honest evaluation:

- windows have the expected columns and ``n_amostras`` features;
- each trial (``source``) belongs to a single unit (no trial spread across units);
- there is exactly one fold per unit, each test is a single unit and every unit is
  tested once;
- in every fold, **no unit and no trial appear at the same time in train and test**
  (the anti-leakage guarantee of the per-group split).

Runs as a script (``python experiments/verificar_janelamento.py``); raises
AssertionError on the first problem and prints a per-fold report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.config import HIPERPARAMETROS  # noqa: E402
from experiments.janelamento import janelar, iter_split_por_unidade  # noqa: E402


def verificar(df_janelado=None) -> list[dict]:
    """Validate the windowing/split invariants; return the per-fold summary."""
    df = df_janelado if df_janelado is not None else janelar()
    assert len(df) > 0, "No windows generated."

    # Columns and number of features
    feature_cols = [c for c in df.columns if c.startswith("massFlow_")]
    n_esperado = HIPERPARAMETROS["n_amostras"]
    assert len(feature_cols) == n_esperado, (
        f"Expected {n_esperado} massFlow_* features, found {len(feature_cols)}"
    )
    for col in ("anomaly", "unit_id", "source"):
        assert col in df.columns, f"Missing column: {col}"

    # No trial spread across more than one unit
    por_source = df.groupby("source")["unit_id"].nunique()
    espalhados = por_source[por_source > 1]
    assert espalhados.empty, f"Trials in more than one unit: {list(espalhados.index)}"

    unidades = set(df["unit_id"].unique())
    resumo = []
    testadas = []

    for unit_teste, df_treino, df_teste in iter_split_por_unidade(df):
        treino_units = set(df_treino["unit_id"].unique())
        teste_units = set(df_teste["unit_id"].unique())

        assert teste_units == {unit_teste}, f"Test should contain only {unit_teste}, contains {teste_units}"
        assert not (treino_units & teste_units), (
            f"Unit leakage in fold {unit_teste}: {treino_units & teste_units}"
        )
        sources_comuns = set(df_treino["source"]) & set(df_teste["source"])
        assert not sources_comuns, f"Trial leakage in fold {unit_teste}: {sources_comuns}"

        testadas.append(unit_teste)
        resumo.append({
            "teste": unit_teste,
            "treino": sorted(treino_units),
            "n_treino": len(df_treino),
            "n_teste": len(df_teste),
        })

    assert len(testadas) == len(unidades), (
        f"Expected {len(unidades)} folds, generated {len(testadas)}"
    )
    assert set(testadas) == unidades, "Not all units were tested exactly once."

    return resumo


if __name__ == "__main__":
    resumo = verificar()
    print("Split verification (no leakage):")
    for info in resumo:
        print(
            f"  test={info['teste']} | train={info['treino']} | "
            f"n_train={info['n_treino']} n_test={info['n_teste']}"
        )
    print(f"\nOK — {len(resumo)} folds, no unit/trial shared between train and test.")
