import pandas as pd

def label_dataset_by_time(
    input_csv=None,      # Modificado: tornamos opcional
    df_dados=None,       # Novo parâmetro para receber DataFrame
    time_ranges=[        # Mantido igual ao seu exemplo original
        (0, 18000, 0),
        (54000, 100000, 1)
    ],
    grey_zone=(18000, 54000),
    exclude_grey=True,
    save_greyzone_csv=False,
    save_csv=True,
    greyzone_csv='greyzone_dataset.csv',
    output_csv='labeled_dataset.csv'
):
    """
    Rotula o dataset de acordo com intervalos de tempo definidos, substituindo a terceira coluna (se existir) pelos rótulos.

    Parâmetros:
    - input_csv (str, opcional): Caminho para o arquivo CSV de entrada.
    - df_dados (DataFrame, opcional): DataFrame diretamente (alternativa a input_csv).
    - time_ranges (list of tuples): Lista de tuplas no formato (start_time, end_time, label).
    - grey_zone (tuple, opcional): Tupla no formato (start_time, end_time) para o espaço cinzento.
    - exclude_grey (bool, opcional): Se True, exclui o espaço cinzento do dataset final.
    - save_greyzone_csv (bool, opcional): Se True, salva os dados da zona cinzenta em um CSV separado.
    - save_csv (bool, opcional): Se True, salva o dataset rotulado em um arquivo CSV.
    - greyzone_csv (str, opcional): Caminho para o arquivo CSV da zona cinzenta.
    - output_csv (str, opcional): Caminho para o arquivo CSV de saída.

    Retorna:
    - df_labeled (DataFrame): Dataset final rotulado.
    - df_grey (DataFrame or None): Dataset da zona cinzenta, se aplicável.
    """

    if df_dados is not None:
        df = df_dados.copy()
    elif input_csv is not None:
        df = pd.read_csv(input_csv)
    else:
        raise ValueError("Forneça input_csv ou df_dados")

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

    df_grey = None

    if grey_zone:
        grey_start, grey_end = grey_zone
        grey_mask = (df['time'] >= grey_start) & (df['time'] <= grey_end)

        if save_greyzone_csv:
            df_grey = df[grey_mask].copy()
            df_grey[label_col] = 'grey_zone'
            df_grey.to_csv(greyzone_csv, index=False)
            print(f"Zona cinzenta salva em {greyzone_csv}")
        elif not exclude_grey:
            df_grey = df[grey_mask].copy()
            df_grey[label_col] = 'grey_zone'

        if exclude_grey:
            df = df[~grey_mask]
        else:
            df.loc[grey_mask, label_col] = 'grey_zone'
    
    if save_csv:
        df.to_csv(output_csv, index=False)
        print(f"Dataset rotulado salvo em {output_csv}")

    return df, df_grey

if __name__ == "__main__":
    
    df1, grey1 = label_dataset_by_time(
        input_csv='dataset_massflow.csv',
        time_ranges=[(0, 18000, 0), (54000, 100000, 1)],
        grey_zone=(18000, 54000),
        exclude_grey=True,
        save_greyzone_csv=False,
        save_csv=True,
        greyzone_csv='greyzone_dataset.csv',
        output_csv='dataset_rotulado.csv'
    )

    
    dados = pd.read_csv('dataset_massflow.csv')  # Carrega antes
    df2, grey2 = label_dataset_by_time(
        df_dados=dados,  # Nova opção
        time_ranges=[(0, 18000, 0), (54000, 100000, 1)],
        grey_zone=(18000, 54000),
        exclude_grey=True,
        save_greyzone_csv=False,
        save_csv=True,
        output_csv='dataset_rotulado.csv'
        
    )