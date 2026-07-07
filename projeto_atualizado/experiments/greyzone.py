"""Inferência na grey zone: probabilidade de amaciamento ao longo do tempo.

Para cada unidade, usa o modelo **fora-da-amostra** (autoencoder + classificador
treinados nas outras 4 unidades) e projeta a série NA completa do compressor —
incluindo a grey zone (18000–54000), que foi excluída do treino/avaliação. A
saída é a probabilidade P(amaciado) por janela ao longo do tempo, evidenciando a
transição do amaciamento sem que o modelo tenha sido informado dela.

Fonte dos dados: os arquivos raw do teste NA (não amaciado) de cada unidade,
descobertos dinamicamente a partir dos ``processado_*_NA``. Como a grey zone não
tem rótulo confiável, a análise é qualitativa (ver docs/ENTENDIMENTO_DADOS.local.md).

Reusa ``janelar``, ``processar_autoencoder``, ``balancear_csv_por_undersampling`` e
``avaliar_modelos_completo``; não altera ``src/``.
"""

import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATASETS_RAW, DATASETS_PROC  # noqa: E402
from experiments.config import HIPERPARAMETROS  # noqa: E402
from experiments.janelamento import janelar  # noqa: E402
from src.analysis.funcao_metodos import avaliar_modelos_completo  # noqa: E402
from src.models.autoencoder import processar_autoencoder  # noqa: E402
from src.preprocessing.funcao_random_undersampling import balancear_csv_por_undersampling  # noqa: E402

CLASSIFICADOR_PADRAO = "regressao_logistica"


def descobrir_arquivos_na_raw(proc_dir=DATASETS_PROC, raw_dir=DATASETS_RAW):
    """Mapeia unidade -> arquivo raw do teste NA, a partir dos ``processado_*_NA``.

    Para cada ``processado_dataset_<A?>_<data>_NA.csv``, procura o raw correspondente
    (``dataset_<A?>_<data>.csv`` ou ``..._NA.csv``).
    """
    mapa = {}
    for p in sorted(proc_dir.glob("processado_dataset_A*_NA.csv")):
        m = re.match(r"processado_dataset_(A\d+)_(.+?)_NA\.csv$", p.name)
        if not m:
            continue
        unit, data = m.group(1), m.group(2)
        for nome in (f"dataset_{unit}_{data}.csv", f"dataset_{unit}_{data}_NA.csv"):
            if (raw_dir / nome).exists():
                mapa[unit] = raw_dir / nome
                break
    return dict(sorted(mapa.items()))


def carregar_teste_na(caminho):
    """Carrega um raw NA (série completa), convertendo vírgula decimal e ordenando por tempo."""
    df = pd.read_csv(caminho)
    for col in ("time", "massFlow"):
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
    return df.sort_values("time").reset_index(drop=True)


def janelar_com_tempo(df, n_amostras, janelamento, amostras_repetidas):
    """Janela ``massFlow`` preservando o tempo da última amostra de cada janela.

    Mesma lógica de passo do ``reorganizar_dataset``, mas mantém a coluna ``time``
    (necessária para plotar a probabilidade ao longo do tempo).
    """
    mass = df["massFlow"].values
    tempo = df["time"].values
    passo = (n_amostras - amostras_repetidas) if janelamento else n_amostras

    linhas, tempos = [], []
    for i in range(0, len(mass) - (n_amostras - 1), passo):
        linhas.append(mass[i:i + n_amostras])
        tempos.append(tempo[i + n_amostras - 1])

    colunas = [f"massFlow_{j + 1}" for j in range(n_amostras)]
    out = pd.DataFrame(linhas, columns=colunas)
    out["time"] = tempos
    return out


def _fixar_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _suavizar(valores, janela):
    """Média móvel centrada (mantém o comprimento)."""
    if janela <= 1:
        return np.asarray(valores, dtype=float)
    s = pd.Series(valores, dtype=float)
    return s.rolling(window=janela, center=True, min_periods=1).mean().values


def probabilidade_por_unidade(hiperparametros=None, seed=42,
                              classificador=CLASSIFICADOR_PADRAO, suavizacao=15):
    """Calcula P(amaciado) ao longo do tempo na série NA de cada unidade.

    Returns:
        dict {unit: DataFrame[time, proba, proba_suave]}, ordenado por tempo.
    """
    hp = {**HIPERPARAMETROS, **(hiperparametros or {})}
    arquivos_na = descobrir_arquivos_na_raw()

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

    curvas = {}
    for unit, caminho_raw in arquivos_na.items():
        _fixar_seed(seed)

        # Treino = as outras 4 unidades, balanceado.
        df_treino = df_janelado[df_janelado["unit_id"] != unit]
        df_treino_bal = balancear_csv_por_undersampling(
            df_dados=df_treino, save_csv=False, embaralhar=True
        )

        # Série NA completa da unidade, janelada com tempo.
        df_na = carregar_teste_na(caminho_raw)
        janelas_na = janelar_com_tempo(
            df_na, hp["n_amostras"], hp["janelamento"], hp["amostras_repetidas"]
        )

        # Treina o AE no treino e projeta as janelas NA no latente.
        with redirect_stdout(io.StringIO()):
            _, df_lat_treino, df_lat_na, _ = processar_autoencoder(
                df_original=df_treino_bal,
                params_autoencoder=params_ae,
                learning_rate=hp["learning_rate"],
                epochs=hp["epochs"],
                batch_size=hp["batch_size"],
                train_size=hp["train_size"],
                df_latente_input=janelas_na,
                return_both_latent=True,
            )

        # Classificador treinado no latente do treino (test dummy = treino).
        _, art = avaliar_modelos_completo(df_lat_treino, df_lat_treino, retornar_modelos=True)
        modelo = art["modelos"][classificador]
        cols = art["feature_cols"]
        X = art["scaler"].transform(df_lat_na[cols].values)
        proba = modelo.predict_proba(X)[:, 1]

        curva = pd.DataFrame({"time": janelas_na["time"].values, "proba": proba})
        curva["proba_suave"] = _suavizar(curva["proba"].values, suavizacao)
        curvas[unit] = curva

    return curvas


if __name__ == "__main__":
    curvas = probabilidade_por_unidade()
    for unit, curva in curvas.items():
        print(f"{unit}: {len(curva)} janelas | tempo [{curva['time'].min():.0f}, {curva['time'].max():.0f}] "
              f"| P inicial~{curva['proba_suave'].iloc[:20].mean():.2f} final~{curva['proba_suave'].iloc[-20:].mean():.2f}")
