"""Central configuration for the run-in detection experiments.

Centralizes, in a single place:
- discovery of the compressor units (A1..A5) and their test files;
- definition of the time ranges used for labeling and of the grey zone;
- the study's fixed hyperparameters;
- output directories.

Nothing here is redundantly hardcoded: the units are discovered from the files in
``datasets/processados/``. The remaining values are the study's working configuration
(adjustable in a single point).
"""

import re
import sys
from pathlib import Path

# Allows ``from paths import ...`` when running from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATASETS_PROC, OUTPUTS_DIR  # noqa: E402

# --- Unit discovery ----------------------------------------------------------
# Each processed file follows the pattern: processado_dataset_A<N>_<date>[_NA].csv
# The unit is the compressor model (A1..A5); each unit may have several tests.
_UNIT_FILE_RE = re.compile(r"^processado_dataset_(A\d+)_.*\.csv$")


def descobrir_unidades(diretorio: Path = DATASETS_PROC) -> dict[str, list[Path]]:
    """Group the processed CSVs by unit (A1..A5).

    Returns an ordered dict {unit_id: [paths of the test files]}.
    """
    unidades: dict[str, list[Path]] = {}
    for caminho in sorted(diretorio.glob("processado_dataset_A*.csv")):
        match = _UNIT_FILE_RE.match(caminho.name)
        if match:
            unidades.setdefault(match.group(1), []).append(caminho)
    return dict(sorted(unidades.items()))


# Unit -> files map, resolved at import time.
UNIDADES = descobrir_unidades()

# --- Time-based labeling -----------------------------------------------------
# (start, end, label): 0 = not run-in, 1 = run-in.
TIME_RANGES = [(0, 18000, 0), (54000, 500000, 1)]
# Transition region, without a reliable label; excluded from training/evaluation.
GREY_ZONE = (18000, 54000)

# --- Study hyperparameters ---------------------------------------------------
# Single working configuration (to be confirmed from the already-executed grid
# search results). Centralized here for reproducibility.
HIPERPARAMETROS = {
    # Preprocessing / windowing
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

# Seeds for the multi-seed runs (statistical evaluation).
SEEDS = [42, 7, 123, 2024, 99]

# --- Outputs -----------------------------------------------------------------
EXPERIMENTS_OUTPUT_DIR = OUTPUTS_DIR / "experiments"


if __name__ == "__main__":
    print("Discovered units:")
    for unit, arquivos in UNIDADES.items():
        print(f"  {unit}: {len(arquivos)} file(s)")
    print(f"\nGrey zone: {GREY_ZONE}")
    print(f"Study output: {EXPERIMENTS_OUTPUT_DIR}")
