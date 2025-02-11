import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def dividir_dados(caminho_arquivo):
    caminho_arquivo = Path(caminho_arquivo)
    
    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    
    # Lê o arquivo Excel, ignorando a primeira linha (header)
    df = pd.read_excel(caminho_arquivo, skiprows=1, header=None)  # Pula a primeira linha
    
    # Obtém apenas a segunda coluna (índice 1)
    dados = df.iloc[:, 1]
    
    # Divide em treino (75%) e validação (25%)
    treino, validacao = train_test_split(dados, test_size=0.25, random_state=42)
    
    return treino, validacao

# Exemplo de uso
caminho = Path(__file__).parent / "meu_arquivo_massflow.xlsx"
treino, validacao = dividir_dados(caminho)
print("Treino:", treino.head())
print("Validação:", validacao.head())
