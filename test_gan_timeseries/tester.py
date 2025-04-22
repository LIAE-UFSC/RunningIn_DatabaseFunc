import pandas as pd

# Carregar o dataset
df = pd.read_csv('arquivo_transformado.csv')

# Excluir as colunas 'signal' e 'unit'
df = df.drop(columns=['signal', 'unit'])

# Converter a coluna 'time' de segundos para horas
df['time_hours'] = df['time'] / 3600

# Preencher a coluna 'anomaly' com base nas condições especificadas
df['anomaly'] = 'amac'  # Padrão para mais de 15 horas
df.loc[df['time_hours'] <= 5, 'anomaly'] = 'n_amac'
df.loc[(df['time_hours'] > 5) & (df['time_hours'] <= 15), 'anomaly'] = 'n_sab'

# Remover a coluna auxiliar 'time_hours' se não for mais necessária
df = df.drop(columns=['time_hours'])

# Salvar o dataset modificado (opcional)
df.to_csv('dataset_modificado.csv', index=False)

# Exibir as primeiras linhas para verificação
print(df.head())