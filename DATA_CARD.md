# Data Card — autoencoder_runin

Description of the dataset used in the run-in detection study.

## Source

Bench tests of hermetic compressors (LIAE-UFSC). Each trial records the **mass flow**
(`massFlow`) over time. Units identified as A1–A5 (compressor model/unit); each unit has
one or more dated trials.

- Signal: `massFlow` (mass flow).
- Sampling: ~1 sample every **60 s** (`time` in seconds: 60, 120, 180, …).
- Variable of interest: the compressor's run-in state.

## Labeling

`anomaly` label: **0 = not run-in**, **1 = run-in** (physical state of the compressor).

- **New** compressor trials (`_NA` files) run in **during** the trial and are labeled
  by time:
  - `t ∈ [0, 18000]` → 0 (not run-in)
  - `t ∈ [54000, ∞)` → 1 (run-in)
- **Already run-in** compressor trials → label 1 for the whole trial.
- **Grey zone** `t ∈ (18000, 54000)`: transition region, **excluded** from
  training/evaluation (ambiguous label). Used only in the qualitative analysis of the
  transition.

Corresponding parameters in `experiments/config.py`:
`TIME_RANGES = [(0, 18000, 0), (54000, 500000, 1)]`, `GREY_ZONE = (18000, 54000)`.

## Counts (processed datasets, grey zone excluded)

| Unit | Trials | Samples | Class 0 (not run-in) | Class 1 (run-in) |
|---|---|---|---|---|
| A1 | 1 | 995 | 299 | 696 |
| A2 | 6 | 8 290 | 299 | 7 991 |
| A3 | 3 | 4 131 | 299 | 3 832 |
| A4 | 4 | 10 494 | 299 | 10 195 |
| A5 | 3 | 6 654 | 299 | 6 355 |
| **Total** | **17** | **30 564** | **1 495** | **29 069** |

Notes:
- All units have **both classes** (each has ≥1 `_NA` trial), which enables the per-unit
  cross-validation in all 5 folds.
- Strong **imbalance** (class 0 is scarce): handled by undersampling in training; the
  evaluation uses imbalance-robust metrics (balanced accuracy, MCC, AUC).

## Reproducibility

- Fixed seeds (`experiments/config.py` → `SEEDS`).
- Counts verifiable via `python experiments/verificar_dados.py`.
