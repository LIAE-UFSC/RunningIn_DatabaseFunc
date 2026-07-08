"""Leakage-free windowing and per-unit (group) split.

Two responsibilities, both aimed at avoiding *data leakage*:

1. ``janelar`` — applies the windowing **per trial** (``source``), reusing
   ``reorganizar_dataset``. Since each trial is windowed in isolation, no window
   crosses the boundary between two trials (see docs/ENTENDIMENTO_DADOS.local.md).
   Each resulting window carries ``unit_id`` and ``source``.

2. ``iter_split_por_unidade`` — yields the train/test splits with the compressor
   unit as the group (scikit-learn's ``LeaveOneGroupOut``). This keeps a whole unit
   on a single side, and overlapping windows never split between train and test.

Does not modify ``src/preprocessing/funcao_janelamento.py``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.config import HIPERPARAMETROS  # noqa: E402
from experiments.data import carregar_combinado  # noqa: E402
from src.preprocessing.funcao_janelamento import reorganizar_dataset  # noqa: E402

COLUNAS_META = ["unit_id", "source"]


def janelar(dados=None, n_amostras=None, janelamento=None, amostras_repetidas=None):
    """Window the data per trial, preserving ``unit_id`` and ``source``.

    Args:
        dados: dict {unit_id: DataFrame}, a combined DataFrame (with ``unit_id`` and
            ``source``) or None (loads via ``carregar_combinado``).
        n_amostras, janelamento, amostras_repetidas: windowing parameters; if None,
            use the values from ``config.HIPERPARAMETROS``.

    Returns:
        DataFrame with columns ``massFlow_1..N``, ``anomaly``, ``unit_id``, ``source``.
        No window mixes different trials.
    """
    n_amostras = HIPERPARAMETROS["n_amostras"] if n_amostras is None else n_amostras
    janelamento = HIPERPARAMETROS["janelamento"] if janelamento is None else janelamento
    amostras_repetidas = (
        HIPERPARAMETROS["amostras_repetidas"] if amostras_repetidas is None else amostras_repetidas
    )

    if dados is None:
        dados = carregar_combinado()
    elif isinstance(dados, dict):
        dados = pd.concat(dados.values(), ignore_index=True)

    janelas = []
    for (unit_id, source), df_ensaio in dados.groupby(COLUNAS_META, sort=False):
        jan = reorganizar_dataset(
            df_dados=df_ensaio.reset_index(drop=True),
            n_amostras=n_amostras,
            janelamento=janelamento,
            amostras_repetidas=amostras_repetidas if janelamento else 1,
            salvar_csv=False,
        )
        jan["unit_id"] = unit_id
        jan["source"] = source
        janelas.append(jan)

    return pd.concat(janelas, ignore_index=True)


def iter_split_por_unidade(df_janelado):
    """Yield the train/test splits leaving one unit out at a time.

    Uses ``LeaveOneGroupOut`` with ``unit_id`` as the group: the test unit never
    appears in the training set, eliminating leakage from overlapping windows.

    Yields:
        (unit_teste, df_treino, df_teste) — DataFrames with a reset index.
    """
    grupos = df_janelado["unit_id"].values
    logo = LeaveOneGroupOut()
    indices = np.arange(len(df_janelado))

    for treino_idx, teste_idx in logo.split(indices, groups=grupos):
        unit_teste = df_janelado.iloc[teste_idx]["unit_id"].iloc[0]
        df_treino = df_janelado.iloc[treino_idx].reset_index(drop=True)
        df_teste = df_janelado.iloc[teste_idx].reset_index(drop=True)
        yield unit_teste, df_treino, df_teste


if __name__ == "__main__":
    df_jan = janelar()
    feature_cols = [c for c in df_jan.columns if c.startswith("massFlow_")]
    print(f"Windows: {len(df_jan)} | features per window: {len(feature_cols)}")
    print("Per unit:")
    for unit_id, grupo in df_jan.groupby("unit_id"):
        print(f"  {unit_id}: {len(grupo):5d} windows | classes {grupo['anomaly'].value_counts().sort_index().to_dict()}")

    print("\nSplits per unit (leave-one-out):")
    for unit_teste, df_treino, df_teste in iter_split_por_unidade(df_jan):
        treino_units = sorted(df_treino["unit_id"].unique())
        print(f"  test={unit_teste} | train={treino_units} | n_train={len(df_treino)} n_test={len(df_teste)}")
