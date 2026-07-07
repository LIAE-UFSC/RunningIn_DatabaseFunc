"""Verificação de ausência de vazamento no janelamento e nos splits.

Confere os invariantes que garantem a avaliação honesta:

- as janelas têm as colunas esperadas e ``n_amostras`` features;
- cada ensaio (``source``) pertence a uma única unidade (nenhum ensaio espalhado);
- há exatamente uma dobra por unidade, cada teste é uma única unidade e todas as
  unidades são testadas uma vez;
- em toda dobra, **nenhuma unidade e nenhum ensaio aparecem ao mesmo tempo em
  treino e teste** (a garantia anti-vazamento do split por grupo).

Roda como script (``python experiments/verificar_janelamento.py``); levanta
AssertionError no primeiro problema e imprime um relatório por dobra.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.config import HIPERPARAMETROS  # noqa: E402
from experiments.janelamento import janelar, iter_split_por_unidade  # noqa: E402


def verificar(df_janelado=None) -> list[dict]:
    """Valida os invariantes do janelamento e dos splits; retorna o resumo por dobra."""
    df = df_janelado if df_janelado is not None else janelar()
    assert len(df) > 0, "Nenhuma janela gerada."

    # Colunas e número de features
    feature_cols = [c for c in df.columns if c.startswith("massFlow_")]
    n_esperado = HIPERPARAMETROS["n_amostras"]
    assert len(feature_cols) == n_esperado, (
        f"Esperadas {n_esperado} features massFlow_*, encontradas {len(feature_cols)}"
    )
    for col in ("anomaly", "unit_id", "source"):
        assert col in df.columns, f"Coluna ausente: {col}"

    # Nenhum ensaio espalhado por mais de uma unidade
    por_source = df.groupby("source")["unit_id"].nunique()
    espalhados = por_source[por_source > 1]
    assert espalhados.empty, f"Ensaios em mais de uma unidade: {list(espalhados.index)}"

    unidades = set(df["unit_id"].unique())
    resumo = []
    testadas = []

    for unit_teste, df_treino, df_teste in iter_split_por_unidade(df):
        treino_units = set(df_treino["unit_id"].unique())
        teste_units = set(df_teste["unit_id"].unique())

        assert teste_units == {unit_teste}, f"Teste deveria conter só {unit_teste}, contém {teste_units}"
        assert not (treino_units & teste_units), (
            f"Vazamento de unidade na dobra {unit_teste}: {treino_units & teste_units}"
        )
        sources_comuns = set(df_treino["source"]) & set(df_teste["source"])
        assert not sources_comuns, f"Vazamento de ensaio na dobra {unit_teste}: {sources_comuns}"

        testadas.append(unit_teste)
        resumo.append({
            "teste": unit_teste,
            "treino": sorted(treino_units),
            "n_treino": len(df_treino),
            "n_teste": len(df_teste),
        })

    assert len(testadas) == len(unidades), (
        f"Esperadas {len(unidades)} dobras, geradas {len(testadas)}"
    )
    assert set(testadas) == unidades, "Nem todas as unidades foram testadas exatamente uma vez."

    return resumo


if __name__ == "__main__":
    resumo = verificar()
    print("Verificação dos splits (sem vazamento):")
    for info in resumo:
        print(
            f"  teste={info['teste']} | treino={info['treino']} | "
            f"n_treino={info['n_treino']} n_teste={info['n_teste']}"
        )
    print(f"\nOK — {len(resumo)} dobras, sem unidade/ensaio compartilhado entre treino e teste.")
