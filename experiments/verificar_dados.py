"""Verification of the expected per-unit data invariants.

Checks, before any experiment, that loading produces data consistent with the
Leave-One-Group-Out scheme:

- expected columns present;
- no missing values in ``time``, ``massFlow`` and ``anomaly``;
- labels only in {0, 1};
- every unit has **both classes** (a prerequisite for LOGO in all folds).

Runs as a script (``python experiments/verificar_dados.py``); raises AssertionError
on the first problem found and prints a per-unit report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.data import carregar_unidades  # noqa: E402

COLUNAS_ESPERADAS = {"time", "massFlow", "anomaly", "unit_id", "source"}
CLASSES_ESPERADAS = {0.0, 1.0}


def verificar(unidades: dict | None = None) -> dict[str, dict]:
    """Validate the invariants and return a summary ``{unit_id: {...}}``.

    Raises ``AssertionError`` with a descriptive message on the first problem.
    """
    dados = unidades if unidades is not None else carregar_unidades()
    assert dados, "No units loaded."

    resumo: dict[str, dict] = {}
    for unit_id, df in dados.items():
        faltando = COLUNAS_ESPERADAS - set(df.columns)
        assert not faltando, f"{unit_id}: missing columns {faltando}"

        for col in ("time", "massFlow", "anomaly"):
            n_na = int(df[col].isna().sum())
            assert n_na == 0, f"{unit_id}: {n_na} missing value(s) in '{col}'"

        classes = set(df["anomaly"].unique().tolist())
        assert classes <= CLASSES_ESPERADAS, f"{unit_id}: unexpected labels {classes - CLASSES_ESPERADAS}"
        assert classes == CLASSES_ESPERADAS, (
            f"{unit_id}: expected classes {CLASSES_ESPERADAS}, found {classes} "
            "(every unit needs both classes for LOGO)"
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
    print("Per-unit data verification:")
    for unit_id, info in resumo.items():
        print(
            f"  {unit_id}: {info['amostras']:5d} samples | "
            f"{info['ensaios']} trial(s) | classes {info['classes']}"
        )
    print(f"\nOK — {len(resumo)} valid units, all with both classes.")
