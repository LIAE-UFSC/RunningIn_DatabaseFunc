import os
import itertools
import pandas as pd
from datetime import datetime
import json
from autoencoderNNpy import BaseModel, Autoencoder, processar_autoencoder, plot_autoencoder_results
from funcao_rotulos import label_dataset_by_time
from funcao_janelamento import reorganizar_dataset
from funcao_random_undersampling import balancear_csv_por_undersampling


def busca_grade_completa(
    input_csv='dataset_massflow.csv',
    # TODOS os parâmetros agora são listas
    lista_time_ranges=[[(0, 18000, 0), (54000, 100000, 1)], 
    ],
    lista_grey_zones=[
        (18000, 54000),  # Padrão
    ],
    lista_n_amostras=[5, 8, 10],
    lista_janelamento=[True, False],
    lista_amostras_repetidas=[1, 4],
    lista_latent_dims=[2, 3, 5],
    lista_learning_rates=[0.01, 0.02, 0.05],
    lista_epochs=[100, 200],
    lista_batch_sizes=[32, 64],
    lista_train_sizes=[0.7, 0.8],
    # Configurações opcionais
    output_dir='resultados_personalizados'
):
    """
    Executa uma busca em grade completa onde TODOS os parâmetros podem variar.
    
    Retorna:
    - Um dicionário com metadados de todas execuções
    - Arquivos salvos em pastas organizadas por combinação
    """
    
    # ==============================================
    # 1. Preparação Inicial
    # ==============================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta_resultados = f"{output_dir}_{timestamp}" if output_dir else f"resultados_{timestamp}"
    os.makedirs(pasta_resultados, exist_ok=True)
    
    # Estrutura para resultados

    metadados = {
        'config': {
            'input_csv': input_csv,
            'total_combinacoes': None,  # Será calculado
            'timestamp': timestamp
        },
        'execucoes': {}
    }

    # ==============================================
    # 2. Gerar TODAS as combinações possíveis
    # ==============================================
    parametros_variados = {

        'time_ranges': lista_time_ranges,
        'grey_zone': lista_grey_zones,
        'n_amostras': lista_n_amostras,
        'janelamento': lista_janelamento,
        'amostras_repetidas': lista_amostras_repetidas,
        'latent_dim': lista_latent_dims,
        'learning_rate': lista_learning_rates,
        'epochs': lista_epochs,
        'batch_size': lista_batch_sizes,
        'train_size': lista_train_sizes
    }

    # Gera todas combinações válidas
    combinacoes = []
    for combo in itertools.product(*parametros_variados.values()):
        current = dict(zip(parametros_variados.keys(), combo))
        
        # Filtra combinações inválidas
        if current['janelamento'] and current['amostras_repetidas'] >= current['n_amostras']:
            continue
            
        combinacoes.append(current)

    metadados['config']['total_combinacoes'] = len(combinacoes)

    # ==============================================
    # 3. Processamento para cada combinação
    # ==============================================

    for i, params in enumerate(combinacoes, 1):
        
        exec_id = f"exec_{i:04d}"
        pasta_exec = os.path.join(pasta_resultados, exec_id)
        os.makedirs(pasta_exec, exist_ok=True)
        
        print(f"\n🔧 Execução {i}/{len(combinacoes)} - ID: {exec_id}")
        
        # --- 3.1 Rotulação Temporal ---
        df_rotulado, _ = label_dataset_by_time(
            
            input_csv=input_csv,
            time_ranges=params['time_ranges'],
            grey_zone=params['grey_zone'],
            exclude_grey=True,
            save_greyzone_csv=False,
            save_csv=False
        )
        
        # --- 3.2 Reorganização ---
        df_reorg = reorganizar_dataset(
            df_dados=df_rotulado,
            n_amostras=params['n_amostras'],
            janelamento=params['janelamento'],
            amostras_repetidas=params['amostras_repetidas'] if params['janelamento'] else None,
            salvar_csv=False
        )
        
        # --- 3.3 Balanceamento (SALVA) ---
        arquivo_balanceado = os.path.join(pasta_exec, 'dataset_balanceado.csv')
        df_balanceado = balancear_csv_por_undersampling(
            df_dados=df_reorg,
            output_csv=arquivo_balanceado,
            embaralhar=False
        )
        
        # --- 3.4 Autoencoder (SALVA latente) ---
        arquivo_latente = os.path.join(pasta_exec, 'espaco_latente.csv')
        _, df_latente, _ = processar_autoencoder(
            df_original=df_balanceado,
            params_autoencoder={
                'input_dim': params['n_amostras'],
                'latent_dim': params['latent_dim']
            },
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            train_size=params['train_size']
        )
        df_latente.to_csv(arquivo_latente, index=False)
        
        # --- 3.5 Registra metadados ---
        metadados['execucoes'][exec_id] = {
            'params': params,
            'arquivos': {
                'balanceado': arquivo_balanceado,
                'latente': arquivo_latente
            }
        }

    # ==============================================
    # 4. Finalização
    # ==============================================

    with open(os.path.join(pasta_resultados, 'metadados_completos.json'), 'w') as f:
        json.dump(metadados, f, indent=4)
    
    print(f"\n Busca concluída! {len(combinacoes)} combinações processadas")
    print(f" Pasta de resultados: {os.path.abspath(pasta_resultados)}")
    
    return metadados


# Exemplo com múltiplas variações
resultados = busca_grade_completa(

    input_csv='dataset_massflow.csv',
    lista_time_ranges=[[(0, 18000, 0), (54000, 100000, 1)],],
    lista_grey_zones=[(18000, 54000),],
    lista_n_amostras=[5, 8],
    lista_janelamento=[True],
    lista_amostras_repetidas=[1, 3],
    lista_latent_dims=[2, 3, 4],
    lista_learning_rates=[0.02],
    lista_epochs=[200],
    lista_batch_sizes=[32],
    lista_train_sizes=[0.7]

)

# Acessar resultados
print(f"Total execuções: {resultados['config']['total_combinacoes']}")
print("Primeira execução:", resultados['execucoes']['exec_0001']['params'])