import pandas as pd
from pathlib import Path

def reorganizar_varios_datasets(
    caminhos_arquivos,        # Lista de caminhos dos arquivos de entrada
    n_amostras=5,
    incluir_tempo=False,
    rotulo_ultimo=True,
    janelamento=False,
    amostras_repetidas=1,
    sufixo_saida="_reorganizado",  # Sufixo para os arquivos de saída
    diretorio_saida=None      # Pasta para salvar os resultados (None = mesma pasta dos inputs)
):
    """
    Processa vários arquivos CSV, reorganiza cada um e salva como novos arquivos CSV.
    
    Parâmetros:
    - caminhos_arquivos: lista de caminhos para os arquivos CSV de entrada
    - n_amostras: número de amostras por linha no output
    - incluir_tempo: se True, inclui a coluna de tempo como primeira feature
    - rotulo_ultimo: se True, usa o rótulo da última amostra do grupo
    - janelamento: se True, cria janelas sobrepostas
    - amostras_repetidas: número de amostras que se repetem entre janelas
    - sufixo_saida: sufixo a ser adicionado aos nomes dos arquivos de saída
    - diretorio_saida: pasta onde salvar os arquivos (None = mesma pasta dos inputs)
    
    Retorna:
    - Lista de tuplas (caminho_entrada, caminho_saida) processados
    """
    
    resultados = []
    
    for caminho in caminhos_arquivos:
        # Carrega o arquivo
        df = pd.read_csv(caminho)
        
        # Processa o DataFrame
        time_values = df['time'].values if incluir_tempo else None
        mass_flow = df['massFlow'].values
        anomaly = df['label'].values if 'label' in df.columns else df['anomaly'].values

        new_data = []
        passo = (n_amostras - amostras_repetidas) if janelamento else n_amostras
        
        for i in range(0, len(mass_flow) - (n_amostras - 1), passo):
            if i + (n_amostras - 1) < len(mass_flow):
                mass_flows = mass_flow[i:i + n_amostras]
                rotulo = anomaly[i + (n_amostras - 1)] if rotulo_ultimo else anomaly[i]
                
                linha = []
                if incluir_tempo:
                    linha.append(time_values[i])
                linha.extend(list(mass_flows))
                linha.append(rotulo)
                
                new_data.append(linha)
        
        # Cria o DataFrame reorganizado
        colunas = []
        if incluir_tempo:
            colunas.append('massFlow_0')
        colunas.extend([f'massFlow_{j+1}' for j in range(n_amostras)])
        colunas.append('anomaly')
        
        new_df = pd.DataFrame(new_data, columns=colunas)
        
        # Define o caminho de saída
        path_entrada = Path(caminho)
        nome_arquivo = f"{path_entrada.stem}{sufixo_saida}{path_entrada.suffix}"
        
        if diretorio_saida:
            path_saida = Path(diretorio_saida) / nome_arquivo
        else:
            path_saida = path_entrada.parent / nome_arquivo
        
        # Salva o arquivo
        new_df.to_csv(path_saida, index=False)
        resultados.append((str(path_entrada), str(path_saida)))
    
    return resultados


# Exemplo de uso:
if __name__ == "__main__":
    # Lista de arquivos para processar
    arquivos = [
        'dados_A1_labeled.csv',
        'dados_A2_labeled.csv',
    ]
    
    # Processa todos os arquivos
    resultados = reorganizar_varios_datasets(
        caminhos_arquivos=arquivos,
        n_amostras=8,
        janelamento=True,
        amostras_repetidas=4,
        sufixo_saida="_reorg"
    )
    
    # Imprime os resultados
    for entrada, saida in resultados:
        print(f"Processado: {entrada} -> {saida}")