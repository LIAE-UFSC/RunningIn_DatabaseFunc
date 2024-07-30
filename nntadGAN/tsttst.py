import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv("C:\\Users\\pedro\\OneDrive\\Área de Trabalho\\meu_arquivo_massflow_A1_csv.csv")

# Exibir as colunas disponíveis
print("Colunas disponíveis:", df.columns)

# Remover as colunas desejadas
colunas_a_remover = ['unit', 'test']  # Substitua pelos nomes das colunas que você deseja remover
df = df.drop(columns=colunas_a_remover)

# Salvar o novo DataFrame em um novo arquivo CSV
df.to_csv('novo_arquivo.csv', index=False)

print("Colunas removidas e novo arquivo salvo como 'novo_arquivo.csv'.")