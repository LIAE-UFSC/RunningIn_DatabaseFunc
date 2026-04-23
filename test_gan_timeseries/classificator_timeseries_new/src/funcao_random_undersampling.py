import pandas as pd

def balancear_csv_por_undersampling(input_csv, coluna_classe='anomaly', output_csv='dataset_balanceado.csv', embaralhar=True):
    """
    Realiza random undersampling em um CSV, balanceando as classes com base na menor classe.

    Parâmetros:
    - input_csv (str): Caminho para o arquivo CSV de entrada.
    - coluna_classe (str): Nome da coluna com os rótulos das classes.
    - output_csv (str): Caminho para salvar o arquivo CSV balanceado.
    - embaralhar (bool): Se True, embaralha as amostras após o balanceamento.

    Retorna:
    - df_balanceado (pd.DataFrame): Dataset balanceado.
    """
    # Carrega o dataset
    df = pd.read_csv(input_csv)

    # Obtém o número de amostras da menor classe
    menor_classe_tamanho = df[coluna_classe].value_counts().min()
    classes = df[coluna_classe].unique()

    # Realiza o undersampling
    amostras_balanceadas = [
        df[df[coluna_classe] == classe].sample(n=menor_classe_tamanho, random_state=42)
        for classe in classes
    ]

    # Concatena os dados balanceados
    df_balanceado = pd.concat(amostras_balanceadas)

    # Embaralha ou ordena
    if embaralhar:
        df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)
    elif 'time' in df.columns:
        df_balanceado = df_balanceado.sort_values(by='time').reset_index(drop=True)
    else:
        df_balanceado = df_balanceado.reset_index(drop=True)

    # Salva o resultado
    df_balanceado.to_csv(output_csv, index=False)
    print(f"Dataset balanceado salvo em {output_csv}")

    return df_balanceado


balancear_csv_por_undersampling(
    input_csv='../data/datasettt_32.csv',
    output_csv='../data/dataset_balanceado.csv',
    embaralhar=False
)

