import pandas as pd

# Caminho do arquivo original
caminho_arquivo = "meu_arquivo_massflow_A1_teste.csv"

# Carrega o CSV
df = pd.read_csv(caminho_arquivo)

# Tempo de 15h em segundos
tempo_15h_em_segundos = 15 * 3600  # 54000 segundos

# Encontra o índice onde os valores de 'anomaly' deixam de ser 1
indice_fim_anomalia_1 = df[df["anomaly"] != 1].index[0]

# Preenche com 2 os valores a partir do fim dos "1" até 15h
df.loc[indice_fim_anomalia_1:, "anomaly"] = df.loc[indice_fim_anomalia_1:, "anomaly"].where(
    df["time"] > tempo_15h_em_segundos, 2
)

# Preenche com 3 os valores depois de 15h
df.loc[df["time"] > tempo_15h_em_segundos, "anomaly"] = 3

# (Opcional) Salva em um novo CSV
df.to_csv("dataset_massflow_A1_com_labels", index=False)