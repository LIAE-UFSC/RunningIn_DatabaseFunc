"""Split the complete dataset into files by unit and test date."""

import pandas as pd
import os
from paths import DATASETS_RAW

def divisao_direta_por_unidade_e_data(caminho_arquivo_csv):
    """
    Read a CSV file and split it into multiple files based on the unique values of
    the 'unit' and 'test' (date) columns.

    The output file name has the format: dataset_{unit}_{DD}_{MM}.csv

    Args:
        caminho_arquivo_csv (str): The path to the complete CSV file.
    """
    # Check that the file exists before continuing
    if not os.path.exists(caminho_arquivo_csv):
        print(f"Error: file '{caminho_arquivo_csv}' not found.")
        return

    # Load the complete dataset
    df_completo = pd.read_csv(caminho_arquivo_csv)

    # Ensure the required columns exist
    if 'unit' not in df_completo.columns or 'test' not in df_completo.columns:
        print("Error: the CSV must contain the 'unit' and 'test' columns.")
        return

    # Iterate over each unique unit
    for unidade in df_completo['unit'].unique():
        # Create a temporary dataframe for the current unit only
        df_unidade = df_completo[df_completo['unit'] == unidade]

        # Iterate over each unique test date for the current unit
        for data_teste in df_unidade['test'].unique():
            # Create the final dataframe for the specific unit and date
            df_final = df_unidade[df_unidade['test'] == data_teste]

            # Format the date for the file name (YYYY_MM_DD -> DD_MM)
            try:
                partes_data = data_teste.split('_')
                ano, mes, dia = partes_data
                data_formatada = f"{dia}_{mes}"
            except (AttributeError, ValueError):
                print(f"Warning: could not format the date '{data_teste}'. Skipping.")
                continue

            # Create the file name and save it
            nome_arquivo = DATASETS_RAW / f"dataset_{unidade}_{data_formatada}.csv"
            df_final.to_csv(nome_arquivo, index=False)
            print(f"File created: {nome_arquivo}")

# --- Example of how to use the function ---
# Simply call the function with your file name.
#
divisao_direta_por_unidade_e_data(str(DATASETS_RAW / 'dataset_completo.csv'))
#
