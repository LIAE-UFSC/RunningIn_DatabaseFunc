"""Grey-zone inference: probability of run-in over time.

For each unit, uses the **out-of-sample** model (autoencoder + classifier trained
on the other 4 units) and projects the unit's full NA series — including the grey
zone (18000–54000), which was excluded from training/evaluation. The output is the
probability P(run-in) per window over time, revealing the run-in transition without
the model having been told about it.

Data source: the raw files of each unit's NA (not run-in) test, discovered
dynamically from the ``processado_*_NA`` files. Since the grey zone has no reliable
label, the analysis is qualitative (see docs/ENTENDIMENTO_DADOS.local.md).

Reuses ``janelar``, ``processar_autoencoder``, ``balancear_csv_por_undersampling`` and
``avaliar_modelos_completo``; does not modify ``src/``.
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
    """Map unit -> raw file of the NA test, from the ``processado_*_NA`` files.

    For each ``processado_dataset_<A?>_<date>_NA.csv``, looks for the matching raw
    file (``dataset_<A?>_<date>.csv`` or ``..._NA.csv``).
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
    """Load a raw NA file (full series), converting the decimal comma and sorting by time."""
    df = pd.read_csv(caminho)
    for col in ("time", "massFlow"):
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
    return df.sort_values("time").reset_index(drop=True)


def janelar_com_tempo(df, n_amostras, janelamento, amostras_repetidas):
    """Window ``massFlow`` preserving the time of the last sample of each window.

    Same stepping logic as ``reorganizar_dataset``, but keeps the ``time`` column
    (needed to plot the probability over time).
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
    """Fix the numpy and torch seeds for fold reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _suavizar(valores, janela):
    """Centered moving average (keeps the length)."""
    if janela <= 1:
        return np.asarray(valores, dtype=float)
    s = pd.Series(valores, dtype=float)
    return s.rolling(window=janela, center=True, min_periods=1).mean().values


def probabilidade_por_unidade(hiperparametros=None, seed=42,
                              classificador=CLASSIFICADOR_PADRAO, suavizacao=15):
    """Compute P(run-in) over time on each unit's NA series.

    Returns:
        dict {unit: DataFrame[time, proba, proba_suave]}, sorted by time.
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

        # Training set = the other 4 units, balanced.
        df_treino = df_janelado[df_janelado["unit_id"] != unit]
        df_treino_bal = balancear_csv_por_undersampling(
            df_dados=df_treino, save_csv=False, embaralhar=True
        )

        # The unit's full NA series, windowed with time.
        df_na = carregar_teste_na(caminho_raw)
        janelas_na = janelar_com_tempo(
            df_na, hp["n_amostras"], hp["janelamento"], hp["amostras_repetidas"]
        )

        # Train the AE on the training set and project the NA windows into the latent.
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

        # Classifier trained on the training latent (dummy test = train).
        _, art = avaliar_modelos_completo(df_lat_treino, df_lat_treino, retornar_modelos=True)
        modelo = art["modelos"][classificador]
        cols = art["feature_cols"]
        X = art["scaler"].transform(df_lat_na[cols].values)
        proba = modelo.predict_proba(X)[:, 1]

        curva = pd.DataFrame({"time": janelas_na["time"].values, "proba": proba})
        curva["proba_suave"] = _suavizar(curva["proba"].values, suavizacao)
        curvas[unit] = curva

    return curvas


def estimar_instante_amaciamento(curva, limiar=0.5):
    """Estimated instant = first time the smoothed P crosses and stays above the threshold.

    Returns:
        float (time) or None if the probability never stays above the threshold.
    """
    p = curva["proba_suave"].values
    t = curva["time"].values
    for i in range(len(p)):
        if np.all(p[i:] >= limiar):
            return float(t[i])
    return None


def estimar_instantes(curvas, limiar=0.5):
    """Apply ``estimar_instante_amaciamento`` to all units."""
    return {unit: estimar_instante_amaciamento(curva, limiar) for unit, curva in curvas.items()}


if __name__ == "__main__":
    curvas = probabilidade_por_unidade()
    instantes = estimar_instantes(curvas)
    for unit, curva in curvas.items():
        inst = instantes[unit]
        inst_str = f"{inst:.0f}" if inst is not None else "—"
        print(f"{unit}: {len(curva)} windows | time [{curva['time'].min():.0f}, {curva['time'].max():.0f}] "
              f"| estimated instant~{inst_str}")
