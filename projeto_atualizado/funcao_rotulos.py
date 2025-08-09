import pandas as pd
import os

def label_dataset_by_time(
    input_csv=None,
    df_dados=None,
    time_ranges=[
        (0, 18000, 0),
        (54000, 100000, 1)
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
    2. Converte a coluna 'time' (com vírgulas) para um formato numérico.
    3. Renomeia a coluna 'test' para 'anomaly'.
    4. Remove a coluna 'unit'.
    5. Aplica os rótulos (ex: 0 para normal, 1 para anomalia) na coluna 'anomaly'
       com base nos intervalos de tempo definidos em 'time_ranges'.
    6. Trata uma "zona cinzenta" (grey_zone), que pode ser excluída ou salva
       separadamente.
    7. Salva o DataFrame processado em um novo arquivo CSV. O nome do arquivo
       de saída é gerado automaticamente a partir do nome do arquivo de entrada
       se não for especificado.

    Parâmetros:
    - input_csv (str, opcional): Caminho para o arquivo CSV de entrada.
    - df_dados (DataFrame, opcional): DataFrame para processar diretamente, como
      alternativa ao input_csv.
    - time_ranges (list of tuples): Lista de tuplas no formato (start_time,
      end_time, label) para definir os rótulos principais.
    - grey_zone (tuple, opcional): Tupla no formato (start_time, end_time) para
      a "zona cinzenta" que pode ser tratada de forma especial.
    - exclude_grey (bool, opcional): Se True, remove os dados da zona cinzenta
      do resultado final. Padrão: True.
    - save_greyzone_csv (bool, opcional): Se True, salva os dados da zona cinzenta
      em um arquivo separado. Padrão: False.
    - save_csv (bool, opcional): Se True, salva o dataset rotulado em um arquivo.
      Padrão: True.
    - greyzone_csv (str, opcional): Nome do arquivo para salvar a zona cinzenta.
      Padrão: 'greyzone_dataset.csv'.
    - output_csv (str, opcional): Nome do arquivo para salvar o resultado final.
      Se None, o nome é gerado automaticamente (ex: 'dataset_rotulado_A1_DD_MM.csv').
      Padrão: None.
    """
    df = None
    df_grey = pd.DataFrame()

    if input_csv:
        df = pd.read_csv(input_csv)
    elif df_dados is not None:
        df = df_dados.copy()
    else:
        raise ValueError("É necessário fornecer 'input_csv' ou 'df_dados'.")

    df['time'] = df['time'].astype(str).str.replace(',', '.').astype(float)

    if 'test' in df.columns:
        df = df.rename(columns={'test': 'anomaly'})
    else:
        if 'anomaly' not in df.columns:
            df['anomaly'] = None
    
    label_col = 'anomaly'
    
    if 'unit' in df.columns:
        df = df.drop(columns=['unit'])

    if save_csv and output_csv is None:
        if input_csv:
            base_name = os.path.splitext(os.path.basename(input_csv))[0]
            output_csv = f"{base_name.replace('dataset', 'dataset_rotulado')}.csv"
        else:
            output_csv = 'dataset_rotulado_default.csv'
            print("Aviso: Nome de saída não especificado para DataFrame. Usando nome padrão.")

    for start_time, end_time, label in time_ranges:
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        df.loc[mask, label_col] = label

    if grey_zone:
        grey_start, grey_end = grey_zone
        grey_mask = (df['time'] >= grey_start) & (df['time'] <= grey_end)

        if save_greyzone_csv:
            df_grey = df[grey_mask].copy()
            df_grey[label_col] = 'grey_zone'
            df_grey.to_csv(greyzone_csv, index=False)
            print(f"Zona cinzenta salva em {greyzone_csv}")
        
        if exclude_grey:
            df = df[~grey_mask]
        else:
            df.loc[grey_mask, label_col] = 'grey_zone'
    
    if save_csv and output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Dataset rotulado salvo em {output_csv}")

    return df, df_grey

if __name__ == "__main__":
    
    print("--- Executando Exemplo 1 (nome de saída automático) ---")
    df1, grey1 = label_dataset_by_time(
        input_csv='dataset_A1_01_07.csv'
    )
    print("Resultado do Exemplo 1:")
    print(df1.head())
    print("-" * 40)

    # print("\n--- Executando Exemplo 2 (nome de saída manual) ---")
    # dados = pd.read_csv('dataset_A2_02_10.csv') 
    # df2, grey2 = label_dataset_by_time(
    #     df_dados=dados,
    #     output_csv='meu_arquivo_customizado.csv'
    # )
