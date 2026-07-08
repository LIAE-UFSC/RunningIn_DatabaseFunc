"""Divisão do dataset completo em arquivos por unidade e data de ensaio."""

import pandas as pd
import os
from paths import DATASETS_RAW

def divisao_direta_por_unidade_e_data(caminho_arquivo_csv):
    """
    Lê um arquivo CSV e o divide em múltiplos arquivos com base nos valores
    únicos das colunas 'unit' e 'test' (data).

    O nome do arquivo de saída terá o formato: dataset_{unidade}_{DD}_{MM}.csv

    Args:
        caminho_arquivo_csv (str): O caminho para o arquivo CSV completo.
    """
    # Verifica se o arquivo existe antes de continuar
    if not os.path.exists(caminho_arquivo_csv):
        print(f"Erro: O arquivo '{caminho_arquivo_csv}' não foi encontrado.")
        return

    # Carrega o dataset completo
    df_completo = pd.read_csv(caminho_arquivo_csv)
    
    # Garante que as colunas necessárias existem
    if 'unit' not in df_completo.columns or 'test' not in df_completo.columns:
        print("Erro: O CSV deve conter as colunas 'unit' e 'test'.")
        return

    # Itera sobre cada unidade única
    for unidade in df_completo['unit'].unique():
        # Cria um dataframe temporário apenas para a unidade atual
        df_unidade = df_completo[df_completo['unit'] == unidade]
        
        # Itera sobre cada data de teste única para a unidade atual
        for data_teste in df_unidade['test'].unique():
            # Cria o dataframe final para a unidade e data específicas
            df_final = df_unidade[df_unidade['test'] == data_teste]
            
            # Formata a data para o nome do arquivo (YYYY_MM_DD -> DD_MM)
            try:
                partes_data = data_teste.split('_')
                ano, mes, dia = partes_data
                data_formatada = f"{dia}_{mes}"
            except (AttributeError, ValueError):
                print(f"Aviso: Não foi possível formatar a data '{data_teste}'. Pulando.")
                continue

            # Cria o nome do arquivo e o salva
            nome_arquivo = DATASETS_RAW / f"dataset_{unidade}_{data_formatada}.csv"
            df_final.to_csv(nome_arquivo, index=False)
            print(f"Arquivo criado: {nome_arquivo}")

# --- Exemplo de como usar a função ---
# Basta chamar a função com o nome do seu arquivo.
#
divisao_direta_por_unidade_e_data(str(DATASETS_RAW / 'dataset_completo.csv'))
#