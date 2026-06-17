"""Baselines de comparação para a representação do autoencoder.

Roda a mesma validação cruzada por unidade do ``runner.py``, mas trocando o
espaço latente do autoencoder por representações de referência:

- **cru**: as próprias janelas de ``massFlow`` (sem nenhuma projeção);
- **PCA**: projeção linear na mesma dimensão do latente (adicionado depois).

O autoencoder só se justifica se superar esses baselines. Reusa os mesmos blocos
(``janelar``, ``iter_split_por_unidade``, ``balancear_csv_por_undersampling``,
``avaliar_modelos_completo``); não altera ``src/`` nem o ``runner.py``.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.config import HIPERPARAMETROS  # noqa: E402
from experiments.janelamento import janelar, iter_split_por_unidade  # noqa: E402
from src.analysis.funcao_metodos import avaliar_modelos_completo  # noqa: E402
from src.preprocessing.funcao_random_undersampling import balancear_csv_por_undersampling  # noqa: E402


def representar_cru(df_treino, df_teste):
    """Representação crua: usa as janelas de ``massFlow`` diretamente (identidade)."""
    cols = [c for c in df_treino.columns if c.startswith("massFlow_")] + ["anomaly"]
    return df_treino[cols].reset_index(drop=True), df_teste[cols].reset_index(drop=True)


def rodar_baseline(representar, df_janelado=None, hiperparametros=None,
                   balancear_treino=True, seed=42, verbose=True, nome="baseline"):
    """Validação cruzada por unidade usando uma representação arbitrária.

    Args:
        representar: callable ``(df_treino_bal, df_teste) -> (rep_treino, rep_teste)``,
            cada saída com features + coluna ``anomaly``. Deve ser ajustada apenas
            no treino e aplicada ao teste (sem vazamento).
        demais args: como em ``runner.rodar_validacao_por_unidade``.

    Returns:
        dict {unit_teste: {classificador: {métricas}}}.
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
            print(f"[{nome} | teste={unit_teste}]")
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
    """Baseline com as janelas de ``massFlow`` cruas."""
    return rodar_baseline(representar_cru, nome="cru", **kwargs)


if __name__ == "__main__":
    rodar_baseline_cru()
