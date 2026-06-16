"""Janelamento sem vazamento e split por unidade (grupo).

Duas responsabilidades, ambas voltadas a evitar *data leakage*:

1. ``janelar`` — aplica o janelamento **por ensaio** (``source``), reusando
   ``reorganizar_dataset``. Como cada ensaio é janelado isoladamente, nenhuma
   janela cruza a fronteira entre dois ensaios (ver docs/ENTENDIMENTO_DADOS.local.md).
   Cada janela resultante carrega ``unit_id`` e ``source``.

2. ``iter_split_por_unidade`` — gera os splits treino/teste com a unidade do
   compressor como grupo (``LeaveOneGroupOut`` do scikit-learn). Assim a unidade
   inteira fica de um único lado, e janelas sobrepostas nunca se dividem entre
   treino e teste.

Não altera ``src/preprocessing/funcao_janelamento.py``.
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
    """Janela os dados por ensaio, preservando ``unit_id`` e ``source``.

    Args:
        dados: dict {unit_id: DataFrame}, DataFrame combinado (com ``unit_id`` e
            ``source``) ou None (carrega via ``carregar_combinado``).
        n_amostras, janelamento, amostras_repetidas: parâmetros do janelamento;
            se None, usam os valores de ``config.HIPERPARAMETROS``.

    Returns:
        DataFrame com colunas ``massFlow_1..N``, ``anomaly``, ``unit_id``, ``source``.
        Nenhuma janela mistura ensaios diferentes.
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
    """Gera os splits treino/teste deixando uma unidade de fora por vez.

    Usa ``LeaveOneGroupOut`` com ``unit_id`` como grupo: a unidade de teste nunca
    aparece no treino, eliminando o vazamento por janelas sobrepostas.

    Yields:
        (unit_teste, df_treino, df_teste) — DataFrames com índice reiniciado.
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
    print(f"Janelas: {len(df_jan)} | features por janela: {len(feature_cols)}")
    print("Por unidade:")
    for unit_id, grupo in df_jan.groupby("unit_id"):
        print(f"  {unit_id}: {len(grupo):5d} janelas | classes {grupo['anomaly'].value_counts().sort_index().to_dict()}")

    print("\nSplits por unidade (deixa-uma-de-fora):")
    for unit_teste, df_treino, df_teste in iter_split_por_unidade(df_jan):
        treino_units = sorted(df_treino["unit_id"].unique())
        print(f"  teste={unit_teste} | treino={treino_units} | n_treino={len(df_treino)} n_teste={len(df_teste)}")
