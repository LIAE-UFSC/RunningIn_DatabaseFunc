"""Per-unit cross-validation of the autoencoder latent space.

For each fold (one unit left out at a time, see ``janelamento.py``):

1. balances the **training set** by undersampling (reuses ``balancear_csv_por_undersampling``);
2. trains the autoencoder on the training set and projects train and test into the
   latent space (reuses ``processar_autoencoder``);
3. trains the classifiers on the training latent and evaluates on the test unit, in
   its real distribution — without balancing the test set (reuses ``avaliar_modelos_completo``).

Balanced training + natural test: the metrics (balanced accuracy, MCC, AUC) already
handle the imbalance, so the test set is not skewed.

Does not modify any module under ``src/``.
"""

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.config import HIPERPARAMETROS, EXPERIMENTS_OUTPUT_DIR  # noqa: E402
from experiments.janelamento import janelar, iter_split_por_unidade  # noqa: E402
from src.analysis.funcao_metodos import avaliar_modelos_completo  # noqa: E402
from src.models.autoencoder import processar_autoencoder  # noqa: E402
from src.preprocessing.funcao_random_undersampling import balancear_csv_por_undersampling  # noqa: E402


def _fixar_seed(seed: int) -> None:
    """Fix the numpy and torch seeds for fold reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def rodar_validacao_por_unidade(df_janelado=None, hiperparametros=None,
                                balancear_treino=True, seed=42, verbose=True,
                                retornar_artefatos=False):
    """Run the leave-one-unit-out cross-validation over the AE latent space.

    Args:
        df_janelado: already-windowed DataFrame (from ``janelar``); if None, windows
            with the effective hyperparameters.
        hiperparametros: dict to override ``config.HIPERPARAMETROS``.
        balancear_treino: if True, undersample the training set before training the AE.
        seed: seed fixed per fold (numpy + torch).
        verbose: prints a per-fold summary.
        retornar_artefatos: if True, also returns, per fold, the latents
            (train/test) and the trained classifiers + scaler, for reuse in the
            figures (UMAP) and in the grey-zone inference.

    Returns:
        - If retornar_artefatos=False: dict {unit_teste: {classifier: {metrics}}}.
        - If retornar_artefatos=True: tuple (resultados, artefatos), with
          artefatos[unit_teste] = {"latente_treino", "latente_teste",
          "classificadores", "scaler"}.
    """
    hp = {**HIPERPARAMETROS, **(hiperparametros or {})}

    if df_janelado is None:
        df_janelado = janelar(
            n_amostras=hp["n_amostras"],
            janelamento=hp["janelamento"],
            amostras_repetidas=hp["amostras_repetidas"],
        )

    params_ae = {
        "input_dim": hp["n_amostras"],
        "latent_dim": hp["latent_dim"],
        "hidden_dim": hp["hidden_dim"],
    }

    resultados = {}
    artefatos = {}
    for unit_teste, df_treino, df_teste in iter_split_por_unidade(df_janelado):
        _fixar_seed(seed)

        df_treino_ae = df_treino
        if balancear_treino:
            df_treino_ae = balancear_csv_por_undersampling(
                df_dados=df_treino, save_csv=False, embaralhar=True
            )

        # Train the AE and project train/test into the latent (silence epoch logs).
        with redirect_stdout(io.StringIO()):
            _, df_lat_treino, df_lat_teste, _ = processar_autoencoder(
                df_original=df_treino_ae,
                params_autoencoder=params_ae,
                learning_rate=hp["learning_rate"],
                epochs=hp["epochs"],
                batch_size=hp["batch_size"],
                train_size=hp["train_size"],
                df_latente_input=df_teste,
                return_both_latent=True,
            )

        if retornar_artefatos:
            metricas, art = avaliar_modelos_completo(
                df_lat_treino, df_lat_teste, retornar_modelos=True
            )
            artefatos[unit_teste] = {
                "latente_treino": df_lat_treino,
                "latente_teste": df_lat_teste,
                "classificadores": art["modelos"],
                "scaler": art["scaler"],
            }
        else:
            metricas = avaliar_modelos_completo(df_lat_treino, df_lat_teste)
        resultados[unit_teste] = metricas

        if verbose:
            print(f"[test={unit_teste}]")
            for nome, m in metricas.items():
                auc = m.get("ROC_AUC")
                auc_str = f"{auc:.3f}" if isinstance(auc, float) else str(auc)
                print(
                    f"    {nome:20s} F1={m.get('F1_score', float('nan')):.3f} "
                    f"balAcc={m.get('Balanced_accuracy', float('nan')):.3f} "
                    f"MCC={m.get('MCC', float('nan')):.3f} AUC={auc_str}"
                )

    if retornar_artefatos:
        return resultados, artefatos
    return resultados


def salvar_metricas(resultados, caminho=None):
    """Serialize the per-fold metrics to JSON. Default: outputs/experiments/."""
    caminho = Path(caminho) if caminho else EXPERIMENTS_OUTPUT_DIR / "metricas_validacao_por_unidade.json"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    return caminho


if __name__ == "__main__":
    resultados = rodar_validacao_por_unidade()
    destino = salvar_metricas(resultados)
    print(f"\nMetrics saved to: {destino}")
