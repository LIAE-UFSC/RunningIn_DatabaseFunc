import pandas as pd

file_path = r"C:\Users\pedro\OneDrive\Área de Trabalho\meu_arquivo_massflow.xlsx"
df = pd.read_excel(file_path)

# Adicionar a coluna "anomaly" se necessário
# df['anomaly'] = 0

# Definir condições para a coluna "anomaly" se necessário
# condition = (df['time'] > 300) & (df['unit'] == 'A1') & (df['test'] == '2019_07_01')
# df.loc[condition, 'anomaly'] = 1

# Salvar o DataFrame atualizado, se necessário
# df.to_excel(r"C:\Users\pedro\OneDrive\Área de Trabalho\meu_arquivo_corrente_thaler_editou_corrente1h.xlsx", index=False)

# Obter todas as unidades únicas
units = df['unit'].unique()

# Salvar um arquivo Excel separado para cada unidade
for unit in units:
    df_unit = df[df['unit'] == unit]
    output_path = rf"C:\Users\pedro\OneDrive\Área de Trabalho\meu_arquivo_massflow_{unit}.xlsx"
    df_unit.to_excel(output_path, index=False)

print("Arquivos salvos com sucesso.")
