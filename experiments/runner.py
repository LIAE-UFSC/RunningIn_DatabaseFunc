"""Validação cruzada por unidade do espaço latente do autoencoder.

Para cada dobra (uma unidade de fora por vez, ver ``janelamento.py``):

1. balanceia o **treino** por undersampling (reusa ``balancear_csv_por_undersampling``);
2. treina o autoencoder no treino e projeta treino e teste no espaço latente
   (reusa ``processar_autoencoder``);
3. treina os classificadores no latente do treino e avalia na unidade de teste,
   em sua distribuição real — sem balancear o teste (reusa ``avaliar_modelos_completo``).

Treino balanceado + teste natural: as métricas (balanced accuracy, MCC, AUC) já
tratam o desbalanceamento, então o teste não é falseado.

Não altera nenhum módulo de ``src/``.
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
    """Fixa as sementes de numpy e torch para reprodutibilidade da dobra."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def rodar_validacao_por_unidade(df_janelado=None, hiperparametros=None,
                                balancear_treino=True, seed=42, verbose=True,
                                retornar_artefatos=False):
    """Roda a validação cruzada deixando-uma-unidade-de-fora sobre o latente do AE.

    Args:
        df_janelado: DataFrame já janelado (de ``janelar``); se None, janela com
            os hiperparâmetros efetivos.
        hiperparametros: dict para sobrepor ``config.HIPERPARAMETROS``.
        balancear_treino: se True, undersampling do treino antes de treinar o AE.
        seed: semente fixada por dobra (numpy + torch).
        verbose: imprime um resumo por dobra.
        retornar_artefatos: se True, também devolve, por dobra, os latentes
            (treino/teste) e os classificadores treinados + scaler, para reuso
            nas figuras (UMAP) e na inferência da grey zone.

    Returns:
        - Se retornar_artefatos=False: dict {unit_teste: {classificador: {métricas}}}.
        - Se retornar_artefatos=True: tupla (resultados, artefatos), com
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

        # Treina o AE e projeta treino/teste no latente (silencia logs de época).
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
            print(f"[teste={unit_teste}]")
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
    """Serializa as métricas por dobra em JSON. Default: outputs/experiments/."""
    caminho = Path(caminho) if caminho else EXPERIMENTS_OUTPUT_DIR / "metricas_validacao_por_unidade.json"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    return caminho


if __name__ == "__main__":
    resultados = rodar_validacao_por_unidade()
    destino = salvar_metricas(resultados)
    print(f"\nMétricas salvas em: {destino}")
