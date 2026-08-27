from pathlib import Path

BASE_DIR = Path(__file__).parent

DATASETS_RAW  = BASE_DIR / "datasets" / "raw"
DATASETS_PROC = BASE_DIR / "datasets" / "processados"
DATASETS_GER  = BASE_DIR / "datasets" / "gerados"

OUTPUTS_DIR   = BASE_DIR / "outputs"
PLOTS_DIR     = OUTPUTS_DIR / "plots"
HEATMAPS_DIR  = OUTPUTS_DIR / "heatmaps"
