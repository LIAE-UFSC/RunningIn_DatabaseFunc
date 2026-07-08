"""Comparison baselines for the autoencoder representation.

Runs the same per-unit cross-validation as ``runner.py``, but replacing the
autoencoder latent space with reference representations:

- **cru** (raw): the ``massFlow`` windows themselves (no projection);
- **PCA**: linear projection to the same dimension as the latent (added later).

The autoencoder is only justified if it beats these baselines. Reuses the same
building blocks (``janelar``, ``iter_split_por_unidade``, ``balancear_csv_por_undersampling``,
``avaliar_modelos_completo``); does not modify ``src/`` nor ``runner.py``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.config import HIPERPARAMETROS  # noqa: E402
from experiments.janelamento import janelar, iter_split_por_unidade  # noqa: E402
from src.analysis.funcao_metodos import avaliar_modelos_completo  # noqa: E402
from src.preprocessing.funcao_random_undersampling import balancear_csv_por_undersampling  # noqa: E402


def representar_cru(df_treino, df_teste):
    """Raw representation: uses the ``massFlow`` windows directly (identity)."""
    cols = [c for c in df_treino.columns if c.startswith("massFlow_")] + ["anomaly"]
    return df_treino[cols].reset_index(drop=True), df_teste[cols].reset_index(drop=True)


def representar_pca(df_treino, df_teste, n_componentes):
    """Linear PCA projection, fit on the training set only and applied to the test.

    Fair comparison with the autoencoder: same input (``massFlow`` windows) and same
    output dimension (``n_componentes`` = ``latent_dim``).
    """
    feat = [c for c in df_treino.columns if c.startswith("massFlow_")]
    pca = PCA(n_components=n_componentes)
    X_treino = pca.fit_transform(df_treino[feat].values)
    X_teste = pca.transform(df_teste[feat].values)

    cols = [f"pca_{i + 1}" for i in range(n_componentes)]
    rep_treino = pd.DataFrame(X_treino, columns=cols)
    rep_treino["anomaly"] = df_treino["anomaly"].values
    rep_teste = pd.DataFrame(X_teste, columns=cols)
    rep_teste["anomaly"] = df_teste["anomaly"].values
    return rep_treino, rep_teste


def rodar_baseline(representar, df_janelado=None, hiperparametros=None,
                   balancear_treino=True, seed=42, verbose=True, nome="baseline"):
    """Per-unit cross-validation using an arbitrary representation.

    Args:
        representar: callable ``(df_treino_bal, df_teste) -> (rep_treino, rep_teste)``,
            each output with features + an ``anomaly`` column. Must be fit only on the
            training set and applied to the test set (no leakage).
        other args: as in ``runner.rodar_validacao_por_unidade``.

    Returns:
        dict {unit_teste: {classifier: {metrics}}}.
    """
    hp = {**HIPERPARAMETROS, **(hiperparametros or {})}

    if df_janelado is None:
        df_janelado = janelar(
            n_amostras=hp["n_amostras"],
            janelamento=hp["janelamento"],
            amostras_repetidas=hp["amostras_repetidas"],
        )

    resultados = {}
    for unit_teste, df_treino, df_teste in iter_split_por_unidade(df_janelado):
        np.random.seed(seed)

        df_treino_bal = df_treino
        if balancear_treino:
            df_treino_bal = balancear_csv_por_undersampling(
                df_dados=df_treino, save_csv=False, embaralhar=True
            )

        rep_treino, rep_teste = representar(df_treino_bal, df_teste)
        metricas = avaliar_modelos_completo(rep_treino, rep_teste)
        resultados[unit_teste] = metricas

        if verbose:
            print(f"[{nome} | test={unit_teste}]")
            for nome_clf, m in metricas.items():
                auc = m.get("ROC_AUC")
                auc_str = f"{auc:.3f}" if isinstance(auc, float) else str(auc)
                print(
                    f"    {nome_clf:20s} F1={m.get('F1_score', float('nan')):.3f} "
                    f"balAcc={m.get('Balanced_accuracy', float('nan')):.3f} "
                    f"MCC={m.get('MCC', float('nan')):.3f} AUC={auc_str}"
                )

    return resultados


def rodar_baseline_cru(**kwargs):
    """Baseline with the raw ``massFlow`` windows."""
    return rodar_baseline(representar_cru, nome="cru", **kwargs)


def rodar_baseline_pca(hiperparametros=None, **kwargs):
    """Baseline with PCA at the same dimension as the latent (``latent_dim``)."""
    hp = {**HIPERPARAMETROS, **(hiperparametros or {})}
    n_componentes = hp["latent_dim"]
    return rodar_baseline(
        lambda df_tr, df_te: representar_pca(df_tr, df_te, n_componentes),
        hiperparametros=hiperparametros,
        nome=f"pca{n_componentes}",
        **kwargs,
    )


if __name__ == "__main__":
    print("=== Baseline: raw features ===")
    rodar_baseline_cru()
    print("\n=== Baseline: PCA (latent_dim) ===")
    rodar_baseline_pca()
