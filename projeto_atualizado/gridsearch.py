import os
import itertools
import pandas as pd
from datetime import datetime
import json
from io import StringIO
from autoencoderNNpy import BaseModel, Autoencoder, processar_autoencoder, plot_autoencoder_results
from funcao_rotulos import label_dataset_by_time
from funcao_janelamento import reorganizar_dataset
from funcao_random_undersampling import balancear_csv_por_undersampling
from funcao_metodos import avaliar_modelos
import numpy as np

np.random.seed(42)  


def busca_grade_completa(
    input_csv='dataset_massflow.csv',
    lista_time_ranges=[[(0, 18000, 0), (54000, 100000, 1)]],
    lista_grey_zones=[(18000, 54000)],
    lista_n_amostras=[5, 8, 10],
    lista_janelamento=[True, False],
    lista_amostras_repetidas=[1, 4],
    lista_latent_dims=[2, 3, 5],
    lista_hidden_dims=[32, 64, 128],  # Novo parâmetro adicionado
    lista_learning_rates=[0.01, 0.02, 0.05],
    lista_epochs=[100, 200],
    lista_batch_sizes=[32, 64],
    lista_train_sizes=[0.7, 0.8],
    output_dir='resultados_personalizados'
):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta_resultados = f"{output_dir}_{timestamp}"
    
    # Cria apenas as pastas para os resultados finais
    pasta_top5_balanceado = os.path.join(pasta_resultados, "top_5percent_balanceado")
    pasta_latente_melhor = os.path.join(pasta_resultados, "latente_melhor")
    pasta_top5_latente = os.path.join(pasta_resultados, "top_5percent_latente")
    
    os.makedirs(pasta_resultados, exist_ok=True)
    os.makedirs(pasta_top5_balanceado, exist_ok=True)
    os.makedirs(pasta_latente_melhor, exist_ok=True)
    os.makedirs(pasta_top5_latente, exist_ok=True)

    metadados = {
        'config': {
            'input_csv': input_csv,
            'total_combinacoes': None,
            'timestamp': timestamp
        },
        'execucoes': {}
    }

    resultados_top5_balanceado = []
    resultados_latente_melhor = []
    resultados_top5_latente = []

    parametros_variados = {

        'time_ranges': lista_time_ranges,
        'grey_zone': lista_grey_zones,
        'n_amostras': lista_n_amostras,
        'janelamento': lista_janelamento,
        'amostras_repetidas': lista_amostras_repetidas,
        'latent_dim': lista_latent_dims,
        'hidden_dim': lista_hidden_dims,
        'learning_rate': lista_learning_rates,
        'epochs': lista_epochs,
        'batch_size': lista_batch_sizes,
        'train_size': lista_train_sizes
    }

    # Geração de todas as combinações possíveis de parâmetros
    combinacoes = []
    for combo in itertools.product(*parametros_variados.values()):
        current = dict(zip(parametros_variados.keys(), combo))
        
        # Restrição 1: Se janelamento=True, amostras_repetidas deve ser < n_amostras
        condicao1 = current['janelamento'] and current['amostras_repetidas'] >= current['n_amostras']
        
        # Restrição 2: hidden_dim deve ser maior que latent_dim (arquitetura do autoencoder)
        condicao2 = current['hidden_dim'] <= current['latent_dim']

        # Restrição 3: hidden_dim deve ser maior que latent_dim (arquitetura do autoencoder)
        condicao3 = current['n_amostras'] <= current['latent_dim']
        
        if condicao1:
            continue
        if condicao2:
            continue
        if condicao3:
            continue

        
        
        combinacoes.append(current)

    metadados['config']['total_combinacoes'] = len(combinacoes)

    for i, params in enumerate(combinacoes, 1):
        exec_id = f"exec_{i:04d}"
        
        print(f"\n🔧 Execução {i}/{len(combinacoes)} - ID: {exec_id}")
        
        # Processamento dos dados - todas as funções configuradas para não salvar arquivos
        df_rotulado, _ = label_dataset_by_time(
            input_csv=input_csv,
            time_ranges=params['time_ranges'],
            grey_zone=params['grey_zone'],
            exclude_grey=True,
            save_greyzone_csv=False,
            save_csv=False
        )
        
        df_reorg = reorganizar_dataset(
            df_dados=df_rotulado,
            n_amostras=params['n_amostras'],
            janelamento=params['janelamento'],
            amostras_repetidas=params['amostras_repetidas'] if params['janelamento'] else None,
            salvar_csv=False
        )
        
        # Balanceamento sem salvar arquivo
        df_balanceado = balancear_csv_por_undersampling(
            df_dados=df_reorg,
            output_csv=None,
            embaralhar=False
        )
        
        # Processamento do autoencoder sem salvar arquivo
        _, df_latente, _ = processar_autoencoder(
            df_original=df_balanceado,
            params_autoencoder={
                'input_dim': params['n_amostras'],
                'latent_dim': params['latent_dim'],
                'hidden_dim': params['hidden_dim']  
            },
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            train_size=params['train_size']
        )
        
        # Avaliação usando arquivos temporários em memória
        with StringIO() as buffer:
            df_balanceado.to_csv(buffer, index=False)
            buffer.seek(0)
            resultados_balanceado = avaliar_modelos(buffer)
        
        with StringIO() as buffer:
            df_latente.to_csv(buffer, index=False)
            buffer.seek(0)
            resultados_latente = avaliar_modelos(buffer)

        # Encontrar melhor classificador para dados balanceados
        melhor_classificador = max(resultados_balanceado.items(), 
                                 key=lambda x: x[1]['Acuracia'])[0]
        melhor_acuracia = resultados_balanceado[melhor_classificador]['Acuracia']

        # Encontrar melhor classificador para dados latentes
        melhor_classificador_latente = max(resultados_latente.items(),
                                         key=lambda x: x[1]['Acuracia'])[0]
        melhor_acuracia_latente = resultados_latente[melhor_classificador_latente]['Acuracia']

        # Salvar para análise do top 5% (dados balanceados)
        resultados_top5_balanceado.append({
            'exec_id': exec_id,
            'classificador': melhor_classificador,
            'dataset': 'balanceado',
            'acuracia': melhor_acuracia,
            'parametros': params
        })

        # Salvar para análise do top 5% (dados latentes)
        resultados_top5_latente.append({
            'exec_id': exec_id,
            'classificador': melhor_classificador_latente,
            'dataset': 'latente',
            'acuracia': melhor_acuracia_latente,
            'parametros': params
        })

        # Verificar se espaço latente foi melhor para algum classificador
        for classificador, metricas in resultados_balanceado.items():
            if resultados_latente[classificador]['Acuracia'] > metricas['Acuracia']:
                resultados_latente_melhor.append({
                    'exec_id': exec_id,
                    'classificador': classificador,
                    'acuracia_balanceado': metricas['Acuracia'],
                    'acuracia_latente': resultados_latente[classificador]['Acuracia'],
                    'parametros': params
                })

        # Atualização dos metadados (sem referências a arquivos)
        metadados['execucoes'][exec_id] = {
            'params': params,
            'resultados_balanceado': resultados_balanceado,
            'resultados_latente': resultados_latente,
            'melhor_classificador': melhor_classificador,
            'melhor_classificador_latente': melhor_classificador_latente
        }

    # Processar top 5% para dados balanceados
    resultados_top5_balanceado.sort(key=lambda x: x['acuracia'], reverse=True)
    num_top5 = max(1, int(len(resultados_top5_balanceado) * 0.05))
    top5_final = resultados_top5_balanceado[:num_top5]
    
    # Processar top 5% para dados latentes
    resultados_top5_latente.sort(key=lambda x: x['acuracia'], reverse=True)
    num_top5_latente = max(1, int(len(resultados_top5_latente) * 0.05))
    top5_latente_final = resultados_top5_latente[:num_top5_latente]
    
    # Salvar resultados consolidados
    with open(os.path.join(pasta_top5_balanceado, 'resultados_top5.json'), 'w', encoding='utf-8') as f:
        json.dump(top5_final, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(pasta_top5_latente, 'resultados_top5_latente.json'), 'w', encoding='utf-8') as f:
        json.dump(top5_latente_final, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(pasta_latente_melhor, 'resultados_latente_melhor.json'), 'w', encoding='utf-8') as f:
        json.dump(resultados_latente_melhor, f, indent=2, ensure_ascii=False)
    
    # Salvar metadados completos
    with open(os.path.join(pasta_resultados, 'metadados_completos.json'), 'w', encoding='utf-8') as f:
        json.dump(metadados, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Busca concluída! {len(combinacoes)} combinações processadas")
    print(f"📁 Pasta de resultados: {os.path.abspath(pasta_resultados)}")
    print(f"📊 Top 5% (balanceado) salvo em: {os.path.join(pasta_top5_balanceado, 'resultados_top5.json')}")
    print(f"📊 Top 5% (latente) salvo em: {os.path.join(pasta_top5_latente, 'resultados_top5_latente.json')}")
    print(f"📈 Casos com latente melhor salvo em: {os.path.join(pasta_latente_melhor, 'resultados_latente_melhor.json')}")
    
    return {

        'pasta_resultados': pasta_resultados,
        'pasta_top5_balanceado': pasta_top5_balanceado,
        'pasta_top5_latente': pasta_top5_latente,
        'pasta_latente_melhor': pasta_latente_melhor
    }


if __name__ == "__main__":
    resultados = busca_grade_completa(
        
        input_csv='dataset_massflow.csv',
        lista_time_ranges=[[(0, 18000, 0), (54000, 100000, 1)]],
        lista_grey_zones=[(18000, 54000)],
        lista_n_amostras=[8, 12, 16],
        lista_janelamento=[True],
        lista_amostras_repetidas=[4, 6],
        lista_latent_dims=[4, 6],
        lista_hidden_dims=[ 64, 128],
        lista_learning_rates=[0.005],
        lista_epochs=[300],
        lista_batch_sizes=[32],
        lista_train_sizes=[0.7],
    )

