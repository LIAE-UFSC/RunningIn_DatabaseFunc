"""Configuração central dos experimentos de detecção de amaciamento (run-in).

Centraliza, em um único lugar:
- descoberta das unidades de compressor (A1..A5) e seus arquivos de teste;
- definição dos intervalos de tempo usados na rotulagem e da grey zone;
- hiperparâmetros fixos do estudo;
- diretórios de saída.

Nada aqui é hardcoded de forma redundante: as unidades são descobertas a partir
dos arquivos em ``datasets/processados/``. Os demais valores são as configurações
de trabalho do estudo (ajustáveis num único ponto).
"""

import re
import sys
from pathlib import Path

# Permite ``from paths import ...`` ao rodar de qualquer lugar.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATASETS_PROC, OUTPUTS_DIR  # noqa: E402

# --- Descoberta das unidades -------------------------------------------------
# Cada arquivo processado tem o padrão: processado_dataset_A<N>_<data>[_NA].csv
# A unidade é o modelo do compressor (A1..A5); cada unidade pode ter vários testes.
_UNIT_FILE_RE = re.compile(r"^processado_dataset_(A\d+)_.*\.csv$")


def descobrir_unidades(diretorio: Path = DATASETS_PROC) -> dict[str, list[Path]]:
    """Agrupa os CSVs processados por unidade (A1..A5).

    Retorna um dict ordenado {unit_id: [caminhos dos arquivos de teste]}.
    """
    unidades: dict[str, list[Path]] = {}
    for caminho in sorted(diretorio.glob("processado_dataset_A*.csv")):
        match = _UNIT_FILE_RE.match(caminho.name)
        if match:
            unidades.setdefault(match.group(1), []).append(caminho)
    return dict(sorted(unidades.items()))


# Mapa unidade -> arquivos, resolvido na importação.
UNIDADES = descobrir_unidades()

# --- Rotulagem temporal ------------------------------------------------------
# (start, end, label): 0 = não amaciado, 1 = amaciado.
TIME_RANGES = [(0, 18000, 0), (54000, 500000, 1)]
# Região de transição, sem rótulo confiável; excluída do treino/avaliação.
GREY_ZONE = (18000, 54000)

# --- Hiperparâmetros do estudo ----------------------------------------------
# Configuração de trabalho única (a ser confirmada a partir dos resultados do
# grid search já executado). Centralizada aqui para reprodutibilidade.
HIPERPARAMETROS = {
    # Pré-processamento / janelamento
    "n_amostras": 8,
    "janelamento": True,
    "amostras_repetidas": 4,
    # Autoencoder
    "latent_dim": 4,
    "hidden_dim": 64,
    "learning_rate": 0.005,
    "epochs": 300,
    "batch_size": 32,
    "train_size": 0.7,
}

# Seeds para as execuções multi-seed (avaliação estatística).
SEEDS = [42, 7, 123, 2024, 99]

# --- Saídas ------------------------------------------------------------------
PAPER_OUTPUT_DIR = OUTPUTS_DIR / "paper"


if __name__ == "__main__":
    print("Unidades descobertas:")
    for unit, arquivos in UNIDADES.items():
        print(f"  {unit}: {len(arquivos)} arquivo(s)")
    print(f"\nGrey zone: {GREY_ZONE}")
    print(f"Saída do estudo: {PAPER_OUTPUT_DIR}")
