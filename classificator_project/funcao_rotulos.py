import pandas as pd

def label_dataset_by_time(
    input_csv,
    time_ranges,
    grey_zone=None,
    exclude_grey=False,
    save_greyzone=False,
    greyzone_csv='greyzone_dataset.csv',
    output_csv='labeled_dataset.csv'
):
    """
    Rotula o dataset de acordo com intervalos de tempo definidos, substituindo a terceira coluna (se existir) pelos rótulos.

    Parâmetros:
    - input_csv (str): Caminho para o arquivo CSV de entrada.
    - time_ranges (list of tuples): Lista de tuplas no formato (start_time, end_time, label).
    - grey_zone (tuple, opcional): Tupla no formato (start_time, end_time) para o espaço cinzento.
    - exclude_grey (bool, opcional): Se True, exclui o espaço cinzento do dataset final.
    - save_greyzone (bool, opcional): Se True, salva os dados da zona cinzenta em um CSV separado.
    - greyzone_csv (str, opcional): Caminho para o arquivo CSV da zona cinzenta.
    - output_csv (str, opcional): Caminho para o arquivo CSV de saída.

    Retorna:
    - None (os datasets são salvos em arquivos CSV).
    """

    df = pd.read_csv(input_csv)
    if 'time' not in df.columns:
        raise ValueError("A coluna 'time' é obrigatória no dataset.")

    if len(df.columns) >= 3:
        label_col = df.columns[2]
    else:
        raise ValueError("O dataset deve conter pelo menos três colunas para substituição.")

    df[label_col] = 'unlabeled'

    for start, end, label in time_ranges:
        mask = (df['time'] >= start) & (df['time'] <= end)
        df.loc[mask, label_col] = label

    if grey_zone:
        grey_start, grey_end = grey_zone
        grey_mask = (df['time'] >= grey_start) & (df['time'] <= grey_end)

        if save_greyzone:
            df_grey = df[grey_mask].copy()
            df_grey[label_col] = 'grey_zone'
            df_grey.to_csv(greyzone_csv, index=False)
            print(f"Zona cinzenta salva em {greyzone_csv}")

        if exclude_grey:
            df = df[~grey_mask]
        else:
            df.loc[grey_mask, label_col] = 'grey_zone'

    df.to_csv(output_csv, index=False)
    print(f"Dataset rotulado salvo em {output_csv}")

time_ranges = [
    (0, 18000, 0),       # Não amaciado
    (54000, 100000, 1),  # Amaciado
]

grey_zone = (18000, 54000)  # Zona de transição (grey zone)

label_dataset_by_time(

    input_csv='dataset_massflow.csv',
    time_ranges=time_ranges,
    grey_zone=grey_zone,
    exclude_grey=True,       
    save_greyzone=True,      #Salva csv da greyzone 
    greyzone_csv='dataset_greyzone.csv',  
    output_csv='dataset_rotulado.csv'

)