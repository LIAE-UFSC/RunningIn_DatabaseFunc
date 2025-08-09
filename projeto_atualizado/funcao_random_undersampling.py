import pandas as pd
import numpy as np

np.random.seed(42)  

def balancear_csv_por_undersampling(
    input_csv=None,      # Modificado: tornamos opcional
    df_dados=None,       # Novo parâmetro para receber DataFrame diretamente
    save_csv=True, 
    coluna_classe='anomaly', 
    output_csv='dataset_balanceado.csv', 
    embaralhar=True
):
    """
    Realiza random undersampling, balanceando as classes com base na menor classe.
    Agora aceita:
    - input_csv (str): Caminho para o arquivo CSV de entrada OU
    - df_dados (DataFrame): DataFrame diretamente

    Parâmetros:
    - input_csv (opcional): str. Caminho para o arquivo CSV de entrada.
    - df_dados (opcional): DataFrame. DataFrame diretamente (alternativa a input_csv).
    - save_csv (bool): Se True, salva o dataset balanceado em um arquivo CSV.
    - coluna_classe (str): Nome da coluna com os rótulos das classes.
    - output_csv (str): Caminho para salvar o arquivo CSV balanceado.
    - embaralhar (bool): Se True, embaralha as amostras após o balanceamento.

    Retorna:
    - df_balanceado (pd.DataFrame): Dataset balanceado.
    """
    # NOVO BLOCO: Seleção da fonte de dados
    if df_dados is not None:
        df = df_dados.copy()
    elif input_csv is not None:
        df = pd.read_csv(input_csv)
    else:
        raise ValueError("Forneça input_csv ou df_dados")

    # Restante da função ORIGINAL (inalterado)
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
        df_balanceado.to_csv(output_csv, index=False)
        print(f"Dataset balanceado salvo em {output_csv}")

    return df_balanceado

# Exemplo de uso com DataFrame (novo)
if __name__ == "__main__":
    # Opção tradicional com arquivo
    balancear_csv_por_undersampling(
        input_csv='dataset_janelado_n_amostras.csv',
        output_csv='dataset_balanceado_pronto.csv',
        embaralhar=False
    )
    
    # Opção nova com DataFrame
    dados = pd.read_csv('dataset_janelado_n_amostras.csv')  # Carrega antes
    
    df_balanceado = balancear_csv_por_undersampling(
        df_dados=dados,  # Novo formato
        output_csv='dataset_balanceado_pronto.csv',
        embaralhar=False
    )