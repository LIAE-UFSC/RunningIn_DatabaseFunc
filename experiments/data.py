"""Loading of the processed datasets, aware of unit and trial.

Each file in ``datasets/processados/`` is a trial of a unit (A1..A5), already
labeled (``anomaly``: 0 = not run-in, 1 = run-in). This layer:

- reads each trial preserving the temporal order;
- attaches ``unit_id`` (compressor model) and ``source`` (originating trial);
- groups the trials by unit.

The ``source`` is essential for leakage-free windowing (Stage 4): windows must not
cross the boundary between two trials. See docs/ENTENDIMENTO_DADOS.local.md.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.config import UNIDADES  # noqa: E402


def carregar_ensaio(caminho: Path, unit_id: str) -> pd.DataFrame:
    """Load a single trial, sorted by time, with ``unit_id`` and ``source``."""
    df = pd.read_csv(caminho).sort_values("time").reset_index(drop=True)
    df["unit_id"] = unit_id
    df["source"] = caminho.stem
    return df


def carregar_unidades(unidades: dict = UNIDADES) -> dict[str, pd.DataFrame]:
    """Return ``{unit_id: DataFrame}`` with each unit's trials concatenated.

    The temporal order is preserved within each trial; distinct trials are stacked
    in sequence (without interleaving), keeping ``source`` to tell them apart.
    """
    return {
        unit_id: pd.concat(
            [carregar_ensaio(p, unit_id) for p in arquivos], ignore_index=True
        )
        for unit_id, arquivos in unidades.items()
    }


def carregar_combinado(unidades: dict = UNIDADES) -> pd.DataFrame:
    """Return a single DataFrame with all units (``unit_id`` column)."""
    return pd.concat(carregar_unidades(unidades).values(), ignore_index=True)


if __name__ == "__main__":
    dados = carregar_unidades()
    print("Loaded units:")
    for unit_id, df in dados.items():
        classes = df["anomaly"].value_counts().sort_index().to_dict()
        n_ensaios = df["source"].nunique()
        print(f"  {unit_id}: {len(df):5d} samples | {n_ensaios} trial(s) | classes {classes}")
    total = sum(len(df) for df in dados.values())
    print(f"\nTotal: {total} samples across {len(dados)} units")
