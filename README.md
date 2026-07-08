# autoencoder_runin — Compressor run-in detection

Pipeline for **run-in detection** of hermetic compressors from mass-flow (`massFlow`)
time series, using an **autoencoder** to learn a latent representation and **classic
classifiers** (logistic regression, SVM-RBF, decision tree) to separate
*not run-in (0)* × *run-in (1)*.

Evaluation uses **per-unit cross-validation** (leave-one-unit-out): the model is always
tested on a compressor that did **not** take part in training, measuring generalization.

## Structure

```
.                     # repository root
  paths.py            # central paths (datasets, outputs)
  gridsearch.py       # exploratory hyperparameter search (original use)
  requirements.txt    # pinned dependencies
  datasets/
    raw/              # raw series per unit/trial
    processados/      # labeled series (not run-in × run-in)
    gerados/          # intermediates
  src/
    preprocessing/    # time labeling, windowing, undersampling
    models/           # autoencoder
    analysis/         # classifier evaluation, UMAP
  experiments/        # evaluation study (see below)
  outputs/
    plots/, heatmaps/ # ablations and curves
    experiments/      # study tables and figures
```

## Study modules (`experiments/`)

| Module | Role |
|---|---|
| `config.py` | units (A1–A5), hyperparameters, grey zone, seeds, paths |
| `data.py` | loads the data per unit (`unit_id`, `source`) |
| `verificar_dados.py` | validates invariants (every unit has both classes) |
| `janelamento.py` | per-trial windowing + per-unit split (leakage-free) |
| `verificar_janelamento.py` | validates no train/test leakage |
| `runner.py` | per-unit cross-validation over the autoencoder latent |
| `baselines.py` | comparison baselines: raw features and PCA |
| `tabela.py` | multi-seed run, aggregation and export of the results table |
| `figuras.py` | per-unit UMAP and grey-zone figures |
| `greyzone.py` | run-in probability over time (grey zone) |

## How to run

Requires the dependencies in `requirements.txt` (an environment with `torch`,
`scikit-learn`, `umap-learn`, `pandas`, `matplotlib`). From the repository root:

```bash
# quick checks
python experiments/verificar_dados.py
python experiments/verificar_janelamento.py

# reproduce the study tables and figures
python experiments/reproduzir.py
```

The hyperparameters live in `experiments/config.py` (`HIPERPARAMETROS`).

## Data

> ⚠️ **Proprietary data.** The run-in dataset belongs to LIAE-UFSC and is
> **not distributed**. It must not be published nor included in any public version
> of this repository.

See [DATA_CARD.md](DATA_CARD.md) for the dataset description (units, per-class counts,
acquisition and label definition).
