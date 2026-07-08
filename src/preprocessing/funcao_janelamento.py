"""Reorganization of the series into windows of N samples (with optional overlap)."""

import pandas as pd
from paths import DATASETS_GER

def reorganizar_dataset(
    caminho_arquivo=None,  # Changed: made optional
    df_dados=None,         # New parameter to receive a DataFrame directly
    n_amostras=5,
    incluir_tempo=False,
    rotulo_ultimo=True,
    salvar_csv=False,
    nome_saida=None,
    janelamento=False,
    amostras_repetidas=1
):
    """
    Reorganize the dataset into groups of 'n_amostras' consecutive samples, with precise overlap control.

    Parameters:
    - caminho_arquivo (optional): str. Path to the original CSV file.
    - df_dados (optional): DataFrame. DataFrame directly (alternative to caminho_arquivo).
    - n_amostras: int. Number of samples per row (default=5).
    - incluir_tempo: bool. If True, adds time as the first feature (massFlow_0).
    - rotulo_ultimo: bool. If True, uses the label of the last sample; otherwise, the first.
    - salvar_csv: bool. If True, saves the DataFrame to a CSV file.
    - nome_saida: str. Output file name (if salvar_csv=True).
    - janelamento: bool. If True, creates overlapping windows.
    - amostras_repetidas: int. How many samples repeat from the previous window (1 <= amostras_repetidas < n_amostras).

    Returns:
    - pandas DataFrame with columns: [massFlow_0 (optional), massFlow_1, ..., massFlow_N, anomaly].
    """
    # Parameter validation
    if janelamento and (amostras_repetidas >= n_amostras or amostras_repetidas < 1):
        raise ValueError("amostras_repetidas must be smaller than n_amostras and greater than or equal to 1")

    if df_dados is not None:
        df = df_dados.copy()
    elif caminho_arquivo is not None:
        df = pd.read_csv(caminho_arquivo)
    else:
        raise ValueError("Provide caminho_arquivo or df_dados")

    time_values = df['time'].values if incluir_tempo else None
    mass_flow = df['massFlow'].values
    anomaly = df['anomaly'].values

    new_data = []

    passo = (n_amostras - amostras_repetidas) if janelamento else n_amostras

    for i in range(0, len(mass_flow) - (n_amostras - 1), passo):
        if i + (n_amostras - 1) < len(mass_flow):

            # Take 'n_amostras' consecutive massFlow values
            mass_flows = mass_flow[i:i + n_amostras]

            # Set the label (last or first of the group)
            rotulo = anomaly[i + (n_amostras - 1)] if rotulo_ultimo else anomaly[i]

            # Build the row: [time_feature (optional), massFlow_1, ..., massFlow_N, label]
            linha = []
            if incluir_tempo:
                linha.append(time_values[i])  # Adds time as massFlow_0
            linha.extend(list(mass_flows))    # Adds massFlow_1 to massFlow_N
            linha.append(rotulo)              # Adds the label

            new_data.append(linha)

    colunas = []
    if incluir_tempo:
        colunas.append('massFlow_0')  # Names time as massFlow_0
    colunas.extend([f'massFlow_{j+1}' for j in range(n_amostras)])
    colunas.append('anomaly')

    new_df = pd.DataFrame(new_data, columns=colunas)

    if salvar_csv:
        caminho_saida = nome_saida if nome_saida is not None else str(DATASETS_GER / 'dataset_reorganizado.csv')
        new_df.to_csv(caminho_saida, index=False)
        print(f"Dataset saved as '{caminho_saida}'")

    return new_df

# Usage example with a DataFrame (new)
if __name__ == "__main__":
    # Traditional option with a file
    df_arquivo = reorganizar_dataset(
        caminho_arquivo=str(DATASETS_GER / 'dataset_rotulado.csv'),
        n_amostras=8,
        janelamento=True,
        salvar_csv=True,
        amostras_repetidas=6
    )

    # New option with a DataFrame
    dados = pd.read_csv(DATASETS_GER / 'dataset_rotulado.csv')  # Load first
    df_direto = reorganizar_dataset(
        df_dados=dados,  # New format
        n_amostras=8,
        janelamento=True,
        amostras_repetidas=4
    )
