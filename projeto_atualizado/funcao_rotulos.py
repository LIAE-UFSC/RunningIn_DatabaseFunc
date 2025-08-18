import pandas as pd
import os

def label_dataset_by_time(
    input_csv=None,
    df_dados=None,
    time_ranges=[
        (0, 18000, 0),
        (54000, 500000, 1)
    ],
    grey_zone=(18000, 54000),
    exclude_grey=True,
    save_greyzone_csv=False,
    save_csv=True,
    greyzone_csv='greyzone_dataset.csv',
    output_csv=None
):
    """
    Processa um dataset: rotula os dados com base no tempo, renomeia a coluna
    'test' para 'anomaly' e remove a coluna 'unit'.

    A função realiza os seguintes passos:
    1. Carrega os dados de um arquivo CSV ou de um DataFrame existente.
    2. Converte TODAS as colunas numéricas (com vírgulas) para float com pontos.
    3. Renomeia a coluna 'test' para 'anomaly'.
    4. Remove a coluna 'unit'.
    5. Aplica os rótulos (ex: 0 para normal, 1 para anomalia) na coluna 'anomaly'
       com base nos intervalos de tempo definidos em 'time_ranges'.
    6. Trata uma "zona cinzenta" (grey_zone), que pode ser excluída ou salva
       separadamente.
    7. Salva o DataFrame processado em um novo arquivo CSV.

    Parâmetros:
    - input_csv (str, opcional): Caminho para o arquivo CSV de entrada.
    - df_dados (DataFrame, opcional): DataFrame para processar diretamente.
    - time_ranges (list of tuples): Lista de tuplas (start_time, end_time, label).
    - grey_zone (tuple, opcional): Tupla (start_time, end_time) para zona cinzenta.
    - exclude_grey (bool): Se True, remove os dados da zona cinzenta.
    - save_greyzone_csv (bool): Se True, salva a zona cinzenta separadamente.
    - save_csv (bool): Se True, salva o dataset rotulado em um arquivo.
    - greyzone_csv (str): Nome do arquivo para a zona cinzenta.
    - output_csv (str, opcional): Nome do arquivo de saída.
    """
    df = None
    df_grey = pd.DataFrame()

    # Carrega os dados
    if input_csv:
        df = pd.read_csv(input_csv)
    elif df_dados is not None:
        df = df_dados.copy()
    else:
        raise ValueError("É necessário fornecer 'input_csv' ou 'df_dados'.")

    # CORREÇÃO PRINCIPAL: Converte todas as colunas numéricas com vírgulas
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        except (ValueError, AttributeError):
            continue  # Mantém colunas não numéricas inalteradas

    # Processamento padrão
    df['time'] = df['time'].astype(str).str.replace(',', '.').astype(float)

    if 'test' in df.columns:
        df = df.rename(columns={'test': 'anomaly'})
    else:
        if 'anomaly' not in df.columns:
            df['anomaly'] = None
    
    if 'unit' in df.columns:
        df = df.drop(columns=['unit'])

    # Gera nome do arquivo de saída se não especificado
    if save_csv and output_csv is None:
        if input_csv:
            base_name = os.path.splitext(os.path.basename(input_csv))[0]
            output_csv = f"{base_name.replace('dataset', 'dataset_rotulado')}.csv"
        else:
            output_csv = 'dataset_rotulado_default.csv'

    # Aplica os rótulos
    for start_time, end_time, label in time_ranges:
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        df.loc[mask, 'anomaly'] = label

    # Trata zona cinzenta
    if grey_zone:
        grey_start, grey_end = grey_zone
        grey_mask = (df['time'] >= grey_start) & (df['time'] <= grey_end)

        if save_greyzone_csv:
            df_grey = df[grey_mask].copy()
            df_grey['anomaly'] = 'grey_zone'
            df_grey.to_csv(greyzone_csv, index=False)
        
        if exclude_grey:
            df = df[~grey_mask]
        else:
            df.loc[grey_mask, 'anomaly'] = 'grey_zone'
    
    # Salva o resultado
    if save_csv and output_csv:
        df.to_csv(output_csv, index=False)

    return df, df_grey

if __name__ == "__main__":

    print("--- Executando Exemplo ---")
    df, grey = label_dataset_by_time(
        input_csv='processado_dataset_A5_22_01_NA.csv',
        output_csv='processado_dataset_A5_22_01_NA.csv'
    )
    print("Dataset processado:")
    print(df.head())

    # pasta = "."

    # arquivos_csv = [f for f in os.listdir(pasta) if f.startswith("dataset_A") and f.endswith(".csv")]

    # for arquivo in arquivos_csv:
    #     print(f"--- Executando para {arquivo} ---")
    #     df, grey = label_dataset_by_time(
    #     input_csv=arquivo,
    #     time_ranges=[
    #         (0, 18000, 0),
    #         (0, 500000, 1)
    #     ],
    #     grey_zone=None,
    #     )
    
    #     # salvar com prefixo
    #     nome_saida = f"processado_{arquivo}"
    #     df.to_csv(nome_saida, index=False)
    
    #     print(f"Dataset processado salvo em: {nome_saida}")
    #     print(df.head())