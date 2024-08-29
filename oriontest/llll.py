import pandas as pd

def remove_columns_from_csv(input_csv_path, output_csv_path):
        
        df = pd.read_csv(input_csv_path)
        
        columns_to_remove = ['unit', 'anomaly', 'signal']
        columns_to_remove = [col for col in columns_to_remove if col in df.columns]
        df.drop(columns=columns_to_remove, inplace=True)
        
        # Salvar o DataFrame modificado em um novo arquivo CSV
        df.to_csv(output_csv_path, index=False)
        print(f"Arquivo salvo com sucesso em: {output_csv_path}")
    

# Exemplo de uso
input_csv = 'C:\\Users\\pedro\\OneDrive\\Documents\\GitHub\\RunningIn_DatabaseFunc\\meu_arquivo_massflow_A1_csv.csv'
output_csv = 'C:\\Users\\pedro\\OneDrive\\Documents\\GitHub\\RunningIn_DatabaseFunc\\oriontest.csv'

remove_columns_from_csv(input_csv, output_csv)
