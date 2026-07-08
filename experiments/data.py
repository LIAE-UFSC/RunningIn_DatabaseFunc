"""Carregamento dos datasets processados, ciente de unidade e ensaio.

Cada arquivo em ``datasets/processados/`` é um ensaio de uma unidade (A1..A5),
já rotulado (``anomaly``: 0 = não amaciado, 1 = amaciado). Esta camada:

- lê cada ensaio preservando a ordem temporal;
- anexa ``unit_id`` (modelo do compressor) e ``source`` (ensaio de origem);
- agrupa os ensaios por unidade.

O ``source`` é essencial para o janelamento sem vazamento (Etapa 4): janelas não
podem cruzar a fronteira entre dois ensaios. Ver docs/ENTENDIMENTO_DADOS.local.md.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.config import UNIDADES  # noqa: E402


def carregar_ensaio(caminho: Path, unit_id: str) -> pd.DataFrame:
    """Carrega um único ensaio, ordenado por tempo, com ``unit_id`` e ``source``."""
    df = pd.read_csv(caminho).sort_values("time").reset_index(drop=True)
    df["unit_id"] = unit_id
    df["source"] = caminho.stem
    return df


def carregar_unidades(unidades: dict = UNIDADES) -> dict[str, pd.DataFrame]:
    """Retorna ``{unit_id: DataFrame}`` com os ensaios de cada unidade concatenados.

    A ordem temporal é preservada dentro de cada ensaio; ensaios distintos são
    empilhados em sequência (sem intercalar), mantendo ``source`` para distingui-los.
    """
    return {
        unit_id: pd.concat(
            [carregar_ensaio(p, unit_id) for p in arquivos], ignore_index=True
        )
        for unit_id, arquivos in unidades.items()
    }


def carregar_combinado(unidades: dict = UNIDADES) -> pd.DataFrame:
    """Retorna um único DataFrame com todas as unidades (coluna ``unit_id``)."""
    return pd.concat(carregar_unidades(unidades).values(), ignore_index=True)


if __name__ == "__main__":
    dados = carregar_unidades()
    print("Unidades carregadas:")
    for unit_id, df in dados.items():
        classes = df["anomaly"].value_counts().sort_index().to_dict()
        n_ensaios = df["source"].nunique()
        print(f"  {unit_id}: {len(df):5d} amostras | {n_ensaios} ensaio(s) | classes {classes}")
    total = sum(len(df) for df in dados.values())
    print(f"\nTotal: {total} amostras em {len(dados)} unidades")
