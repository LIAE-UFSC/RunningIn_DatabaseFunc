"""Class balancing by random undersampling (matches the smallest class)."""

import pandas as pd
import numpy as np
from paths import DATASETS_GER

np.random.seed(42)

def balancear_csv_por_undersampling(
    input_csv=None,      # Changed: made optional
    df_dados=None,       # New parameter to receive a DataFrame directly
    save_csv=True,
    coluna_classe='anomaly',
    output_csv=None,
    embaralhar=True
):
    """
    Performs random undersampling, balancing the classes based on the smallest class.
    Now accepts:
    - input_csv (str): Path to the input CSV file OR
    - df_dados (DataFrame): DataFrame directly

    Parameters:
    - input_csv (optional): str. Path to the input CSV file.
    - df_dados (optional): DataFrame. DataFrame directly (alternative to input_csv).
    - save_csv (bool): If True, saves the balanced dataset to a CSV file.
    - coluna_classe (str): Name of the column with the class labels.
    - output_csv (str): Path to save the balanced CSV file.
    - embaralhar (bool): If True, shuffles the samples after balancing.

    Returns:
    - df_balanceado (pd.DataFrame): Balanced dataset.
    """
    # NEW BLOCK: data source selection
    if df_dados is not None:
        df = df_dados.copy()
    elif input_csv is not None:
        df = pd.read_csv(input_csv)
    else:
        raise ValueError("Provide input_csv or df_dados")

    # Rest of the ORIGINAL function (unchanged)
    menor_classe_tamanho = df[coluna_classe].value_counts().min()
    classes = df[coluna_classe].unique()

    amostras_balanceadas = [
        df[df[coluna_classe] == classe].sample(n=menor_classe_tamanho, random_state=42)
        for classe in classes
    ]

    df_balanceado = pd.concat(amostras_balanceadas)

    if embaralhar:
        df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)
    elif 'time' in df.columns:
        df_balanceado = df_balanceado.sort_values(by='time').reset_index(drop=True)
    else:
        df_balanceado = df_balanceado.reset_index(drop=True)

    if save_csv:
        caminho_saida = output_csv if output_csv is not None else str(DATASETS_GER / 'dataset_balanceado.csv')
        df_balanceado.to_csv(caminho_saida, index=False)
        print(f"Balanced dataset saved to {caminho_saida}")

    return df_balanceado

# Usage example with a DataFrame (new)
if __name__ == "__main__":
    # Traditional option with a file
    balancear_csv_por_undersampling(
        input_csv=str(DATASETS_GER / 'dataset_reorganizado.csv'),
        output_csv=str(DATASETS_GER / 'dataset_balanceado_pronto.csv'),
        embaralhar=False
    )

    # New option with a DataFrame
    dados = pd.read_csv(DATASETS_GER / 'dataset_janelado_n_amostras.csv')  # Load first

    df_balanceado = balancear_csv_por_undersampling(
        df_dados=dados,  # New format
        output_csv=str(DATASETS_GER / 'dataset_balanceado_pronto.csv'),
        embaralhar=False
    )
