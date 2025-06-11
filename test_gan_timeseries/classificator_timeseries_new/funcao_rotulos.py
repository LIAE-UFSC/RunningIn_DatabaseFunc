import pandas as pd

def label_dataset_by_time(input_csv, time_ranges, grey_zone=None, exclude_grey=False, output_csv='labeled_dataset.csv'):
    """
    Rotula o dataset de acordo com intervalos de tempo definidos, substituindo a terceira coluna (se existir) pelos rótulos.

    Parâmetros:
    - input_csv (str): Caminho para o arquivo CSV de entrada.
    - time_ranges (list of tuples): Lista de tuplas no formato (start_time, end_time, label).
    - grey_zone (tuple, opcional): Tupla no formato (start_time, end_time) para o espaço cinzento.
    - exclude_grey (bool, opcional): Se True, exclui o espaço cinzento do dataset final.
    - output_csv (str, opcional): Caminho para o arquivo CSV de saída.

    Retorna:
    - None (o dataset rotulado é salvo em output_csv).
    """
    # Carrega o dataset
    df = pd.read_csv(input_csv)

    # Verifica se a coluna 'time' existe
    if 'time' not in df.columns:
        raise ValueError("A coluna 'time' é obrigatória no dataset.")

    # Determina a coluna alvo (a terceira coluna, índice 2)
    if len(df.columns) >= 3:
        label_col = df.columns[2]
    else:
        raise ValueError("O dataset deve conter pelo menos três colunas para substituição.")

    # Inicializa a coluna com 'unlabeled'
    df[label_col] = 'unlabeled'

    # Aplica os rótulos de acordo com os intervalos de tempo
    for start, end, label in time_ranges:
        mask = (df['time'] >= start) & (df['time'] <= end)
        df.loc[mask, label_col] = label

    # Trata o espaço cinzento
    if grey_zone:
        grey_start, grey_end = grey_zone
        grey_mask = (df['time'] >= grey_start) & (df['time'] <= grey_end)
        if exclude_grey:
            df = df[~grey_mask]
        else:
            df.loc[grey_mask, label_col] = 'grey_zone'

    # Salva o dataset rotulado
    df.to_csv(output_csv, index=False)
    print(f"Dataset rotulado salvo em {output_csv}")


 
time_ranges = [
    
    (0, 18000, 0), #intervalo de tempo, rotulo
    (54000, 100000, 1),
]

grey_zone = (18000, 54000)

label_dataset_by_time(
    input_csv='dataset_modificado.csv',
    time_ranges=time_ranges,
    grey_zone=grey_zone,
    exclude_grey=True,
    output_csv='dataset_rotulado.csv'
)

