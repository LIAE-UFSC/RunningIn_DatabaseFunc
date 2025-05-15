import pandas as pd

def reorganizar_dataset(
    caminho_arquivo, 
    n_amostras=5, 
    incluir_tempo=False,
    rotulo_ultimo=True,
    salvar_csv=False,
    nome_saida='dataset_reorganizado.csv',
    janelamento=False,
    amostras_repetidas=1
):
    """
    Reorganiza o dataset em grupos de 'n_amostras' consecutivas, com controle preciso de overlap.

    Parâmetros:
    - caminho_arquivo: str. Caminho do arquivo CSV original.
    - n_amostras: int. Quantidade de amostras por linha (padrão=5).
    - incluir_tempo: bool. Se True, adiciona o tempo como primeira feature (massFlow_0).
    - rotulo_ultimo: bool. Se True, usa o rótulo da última amostra; senão, usa o da primeira.
    - salvar_csv: bool. Se True, salva o DataFrame em um arquivo CSV.
    - nome_saida: str. Nome do arquivo de saída (se salvar_csv=True).
    - janelamento: bool. Se True, cria janelas sobrepostas.
    - amostras_repetidas: int. Quantas amostras devem se repetir da janela anterior (1 <= amostras_repetidas < n_amostras).

    Retorna:
    - DataFrame pandas com colunas: [massFlow_0 (opcional), massFlow_1, ..., massFlow_N, anomaly].
    """
    # Validação dos parâmetros
    if janelamento and (amostras_repetidas >= n_amostras or amostras_repetidas < 1):
        raise ValueError("amostras_repetidas deve ser menor que n_amostras e maior ou igual a 1")
    
    # Carrega o dataset
    df = pd.read_csv(caminho_arquivo)
    
    # Extrai colunas
    time_values = df['time'].values if incluir_tempo else None
    mass_flow = df['massFlow'].values
    anomaly = df['anomaly'].values

    # Prepara lista de dados reorganizados
    new_data = []
    
    # Define o passo para a iteração
    passo = (n_amostras - amostras_repetidas) if janelamento else n_amostras
    
    for i in range(0, len(mass_flow) - (n_amostras - 1), passo):
        if i + (n_amostras - 1) < len(mass_flow):
            # Pega 'n_amostras' valores consecutivos de massFlow
            mass_flows = mass_flow[i:i + n_amostras]
            
            # Define o rótulo (último ou primeiro do grupo)
            rotulo = anomaly[i + (n_amostras - 1)] if rotulo_ultimo else anomaly[i]
            
            # Monta a linha: [time_feature (opcional), massFlow_1, ..., massFlow_N, rotulo]
            linha = []
            if incluir_tempo:
                linha.append(time_values[i])  # Adiciona tempo como massFlow_0
            linha.extend(list(mass_flows))    # Adiciona massFlow_1 a massFlow_N
            linha.append(rotulo)              # Adiciona rótulo
            
            new_data.append(linha)
    
    # Define nomes das colunas
    colunas = []
    if incluir_tempo:
        colunas.append('massFlow_0')  # Nomeia o tempo como massFlow_0
    colunas.extend([f'massFlow_{j+1}' for j in range(n_amostras)])
    colunas.append('anomaly')
    
    # Cria o DataFrame
    new_df = pd.DataFrame(new_data, columns=colunas)
    
    # Salva em CSV se solicitado
    if salvar_csv:
        new_df.to_csv(nome_saida, index=False)
        print(f"Dataset salvo como '{nome_saida}'")
    
    return new_df


# Exemplo de uso:
df_com_tempo = reorganizar_dataset(

    caminho_arquivo='dataset_modificado.csv',
    n_amostras=7,
    incluir_tempo=False,
    salvar_csv=True,
    nome_saida='dataset_com_tempo____teste_9090.csv',
    janelamento=False,
    amostras_repetidas=6
)