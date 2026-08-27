import pandas as pd
import os
from typing import List

def concatenar_datasets(
    arquivos_entrada: List[str],  # Lista de caminhos dos arquivos CSV
    arquivo_saida: str = "dataset_concatenado.csv",  # Caminho de saída
    ordenar_por: str = None,      # Coluna para ordenação (opcional, ex: 'time')
    remover_duplicados: bool = True  # Remover linhas duplicadas
) -> pd.DataFrame:
    """
    Concatena múltiplos arquivos CSV em um único DataFrame e salva o resultado.

    Parâmetros:
    - arquivos_entrada: Lista de caminhos para os arquivos CSV.
    - arquivo_saida: Caminho onde o arquivo concatenado será salvo.
    - ordenar_por: Coluna para ordenar o DataFrame final (opcional).
    - remover_duplicados: Se True, remove linhas duplicadas.

    Retorna:
    - DataFrame concatenado.
    """
    if not arquivos_entrada:
        raise ValueError("A lista de arquivos de entrada está vazia.")

    # Carregar e concatenar todos os DataFrames
    dfs = []
    for arquivo in arquivos_entrada:
        try:
            df = pd.read_csv(arquivo)
            dfs.append(df)
            print(f"Arquivo carregado: {arquivo} ({len(df)} linhas)")
        except Exception as e:
            print(f"Erro ao carregar {arquivo}: {str(e)}")
            continue

    if not dfs:
        raise ValueError("Nenhum DataFrame válido foi carregado.")

    df_concatenado = pd.concat(dfs, ignore_index=True)

    # Remover duplicados (se habilitado)
    if remover_duplicados:
        tamanho_inicial = len(df_concatenado)
        df_concatenado = df_concatenado.drop_duplicates()
        duplicados_removidos = tamanho_inicial - len(df_concatenado)
        print(f"Duplicados removidos: {duplicados_removidos}")

    # Ordenar (se especificado)
    if ordenar_por and ordenar_por in df_concatenado.columns:
        df_concatenado = df_concatenado.sort_values(ordenar_por)

    # Salvar em CSV
    df_concatenado.to_csv(arquivo_saida, index=False)
    print(f"\nDataset concatenado salvo em: {arquivo_saida}")
    print(f"Total de registros: {len(df_concatenado)}")

    return df_concatenado


# Exemplo de uso
if __name__ == "__main__":
    # Lista dos arquivos já processados (rotulados/reorganizados)
    arquivos = [
        "dados_A1_labeled_reorg.csv",
        "dados_A2_labeled_reorg.csv",
    ]

    # Concatenar e salvar
    df_final = concatenar_datasets(
        arquivos_entrada=arquivos,
        arquivo_saida="dados_concatenados_finais.csv",
        ordenar_por="time",          # Opcional: ordenar por tempo
        remover_duplicados=True      # Remover linhas repetidas
    )