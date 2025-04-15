
import pandas as pd

def transformar_anomalias(input_file, output_file):
    """
    Transforma as classes de anomalia (1,2,3) em (A,B,C) em um arquivo CSV.
    
    Parâmetros:
    input_file (str): Caminho do arquivo CSV de entrada
    output_file (str): Caminho do arquivo CSV de saída
    """
    try:
        # Ler o arquivo CSV
        df = pd.read_csv(input_file)
        
        # Verificar se a coluna 'anomaly' existe
        if 'anomaly' not in df.columns:
            raise ValueError("O arquivo CSV não contém uma coluna 'anomaly'")
        
        # Mapeamento das classes
        anomaly_map = {1: 'A', 2: 'B', 3: 'C'}
        
        # Aplicar a transformação
        df['anomaly'] = df['anomaly'].map(anomaly_map)
        
        # Salvar o resultado
        df.to_csv(output_file, index=False)
        
        print(f"Transformação concluída! Resultado salvo em: {output_file}")
        print("\nPrimeiras linhas do arquivo transformado:")
        print(df.head())
        
        return True
    
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado - {input_file}")
        return False
    except Exception as e:
        print(f"Ocorreu um erro: {str(e)}")
        return False

# Exemplo de uso (substitua pelos seus caminhos reais)
if __name__ == "__main__":
    input_csv = caminho_arquivo = r'C:\Users\PC-1\Documents\GitHub\RunningIn_DatabaseFunc\test_gan_timeseries\dataset_massflow_A1_com_labels.csv'
    output_csv = 'arquivo_transformadoss.csv'  # Nome do arquivo de saída
    
    transformar_anomalias(input_csv, output_csv)