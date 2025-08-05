import csv
from collections import defaultdict

def separar_por_unidade(input_file):
    # Dicionário para armazenar os dados por unidade
    dados_por_unidade = defaultdict(list)
    
    # Lê o arquivo CSV
    with open(input_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        # Armazena o cabeçalho
        cabecalho = reader.fieldnames
        
        # Separa os dados por unidade
        for linha in reader:
            unidade = linha['unit']
            dados_por_unidade[unidade].append(linha)
    
    # Salva cada grupo em um arquivo CSV separado
    for unidade, dados in dados_por_unidade.items():
        output_file = f'dados_{unidade}.csv'
        with open(output_file, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=cabecalho)
            writer.writeheader()
            writer.writerows(dados)
        
        print(f'Arquivo salvo: {output_file} com {len(dados)} registros')
    
    return len(dados_por_unidade)

# Nome do arquivo de entrada
input_file = 'big_dataset.csv'

# Executa a separação
num_arquivos = separar_por_unidade(input_file)
print(f'\nTotal de arquivos criados: {num_arquivos}')