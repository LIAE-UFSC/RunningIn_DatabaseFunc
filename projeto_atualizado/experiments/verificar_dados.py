"""Verificação dos invariantes esperados dos dados por unidade.

Confere, antes de qualquer experimento, que o carregamento produz dados
consistentes para o Leave-One-Group-Out:

- colunas esperadas presentes;
- sem valores ausentes em ``time``, ``massFlow`` e ``anomaly``;
- rótulos apenas em {0, 1};
- toda unidade tem **as duas classes** (pré-requisito do LOGO em todas as dobras).

Roda como script (``python experiments/verificar_dados.py``); levanta AssertionError
no primeiro problema encontrado e imprime um relatório por unidade.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.data import carregar_unidades  # noqa: E402

COLUNAS_ESPERADAS = {"time", "massFlow", "anomaly", "unit_id", "source"}
CLASSES_ESPERADAS = {0.0, 1.0}


def verificar(unidades: dict | None = None) -> dict[str, dict]:
    """Valida os invariantes e retorna um resumo ``{unit_id: {...}}``.

    Levanta ``AssertionError`` com mensagem descritiva no primeiro problema.
    """
    dados = unidades if unidades is not None else carregar_unidades()
    assert dados, "Nenhuma unidade carregada."

    resumo: dict[str, dict] = {}
    for unit_id, df in dados.items():
        faltando = COLUNAS_ESPERADAS - set(df.columns)
        assert not faltando, f"{unit_id}: colunas ausentes {faltando}"

        for col in ("time", "massFlow", "anomaly"):
            n_na = int(df[col].isna().sum())
            assert n_na == 0, f"{unit_id}: {n_na} valor(es) ausente(s) em '{col}'"

        classes = set(df["anomaly"].unique().tolist())
        assert classes <= CLASSES_ESPERADAS, f"{unit_id}: rótulos inesperados {classes - CLASSES_ESPERADAS}"
        assert classes == CLASSES_ESPERADAS, (
            f"{unit_id}: esperadas as classes {CLASSES_ESPERADAS}, encontradas {classes} "
            "(toda unidade precisa das duas classes para o LOGO)"
        )

        contagem = df["anomaly"].value_counts().sort_index().to_dict()
        resumo[unit_id] = {
            "amostras": len(df),
            "ensaios": int(df["source"].nunique()),
            "classes": contagem,
        }
    return resumo


if __name__ == "__main__":
    resumo = verificar()
    print("Verificação dos dados por unidade:")
    for unit_id, info in resumo.items():
        print(
            f"  {unit_id}: {info['amostras']:5d} amostras | "
            f"{info['ensaios']} ensaio(s) | classes {info['classes']}"
        )
    print(f"\nOK — {len(resumo)} unidades válidas, todas com as duas classes.")
