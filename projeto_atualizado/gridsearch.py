import os
import numpy as np
import itertools
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
from io import StringIO

RESULTS_DIR = Path(__file__).parent.parent / "resultados"
from src.models.autoencoder import BaseModel, Autoencoder, processar_autoencoder, plot_autoencoder_results
from src.preprocessing.funcao_rotulos import label_dataset_by_time
from src.preprocessing.funcao_janelamento import reorganizar_dataset
from src.preprocessing.funcao_random_undersampling import balancear_csv_por_undersampling
from src.analysis.funcao_metodos import avaliar_modelos
from paths import DATASETS_RAW, DATASETS_PROC

np.random.seed(42)

def busca_grade_completa(
    input_csvs=[str(DATASETS_RAW / 'dataset_massflow.csv')],
    test_csv=None,  
    lista_time_ranges=[[(0, 18000, 0), (54000, 100000, 1)]],
    lista_grey_zones=[(18000, 54000)],
    lista_n_amostras=[5, 8, 10],
    lista_janelamento=[True, False],
    lista_amostras_repetidas=[1, 4],
    lista_latent_dims=[5],
    lista_hidden_dims=[32, 128],
    lista_learning_rates=[0.01, 0.05],
    lista_epochs=[100, 200],
    lista_batch_sizes=[32],
    lista_train_sizes=[0.7],
    output_dir=None
):
    base = Path(output_dir) if output_dir else RESULTS_DIR / "resultados_personalizados"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta_resultados = str(base.parent / f"{base.name}_{timestamp}")
    
    pasta_top5_balanceado = os.path.join(pasta_resultados, "top_5percent_balanceado")
    pasta_latente_melhor = os.path.join(pasta_resultados, "latente_melhor")
    pasta_top5_latente = os.path.join(pasta_resultados, "top_5percent_latente")
    
    os.makedirs(pasta_resultados, exist_ok=True)
    os.makedirs(pasta_top5_balanceado, exist_ok=True)
    os.makedirs(pasta_latente_melhor, exist_ok=True)
    os.makedirs(pasta_top5_latente, exist_ok=True)

    metadados = {
        'config': {
            'input_csvs': input_csvs,
            'test_csv': test_csv,
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

    combinacoes = []
    for combo in itertools.product(*parametros_variados.values()):
        current = dict(zip(parametros_variados.keys(), combo))
        
        # Restrições 
        condicao1 = current['janelamento'] and current['amostras_repetidas'] >= current['n_amostras']
        condicao2 = current['hidden_dim'] <= current['latent_dim']
        condicao3 = current['n_amostras'] <= current['latent_dim']
        
        if condicao1 or condicao2 or condicao3:
            continue
            
        combinacoes.append(current)

    metadados['config']['total_combinacoes'] = len(combinacoes)

    for i, params in enumerate(combinacoes, 1):
        exec_id = f"exec_{i:04d}"
        
        print(f"\n🔧 Execução {i}/{len(combinacoes)} - ID: {exec_id}")
        
        # Processa cada arquivo de input
        dfs_processed = []
        for csv_path in input_csvs:  # input_csvs é uma lista de caminhos
            # Carrega o CSV já rotulado
            df = pd.read_csv(csv_path)
            
            # Reorganiza o dataset
            df_reorg = reorganizar_dataset(
                df_dados=df,
                n_amostras=params['n_amostras'],
                janelamento=params['janelamento'],
                amostras_repetidas=params['amostras_repetidas'] if params['janelamento'] else None,
                salvar_csv=False
            )
            dfs_processed.append(df_reorg)
        
        
        df_combined = pd.concat(dfs_processed, ignore_index=True)
        
        
        df_test_balanceado = None
        
        if test_csv:
            # Se test_csv for uma lista de arquivos
            if isinstance(test_csv, list):
                dfs_test = []
                for test_path in test_csv:
                    df_test = pd.read_csv(test_path)
                    df_test_reorg = reorganizar_dataset(
                        df_dados=df_test,
                        n_amostras=params['n_amostras'],
                        janelamento=params['janelamento'],
                        amostras_repetidas=params['amostras_repetidas'] if params['janelamento'] else None,
                        salvar_csv=False
                    )
                    dfs_test.append(df_test_reorg)
                df_test_combined = pd.concat(dfs_test, ignore_index=True)
                df_test_balanceado = balancear_csv_por_undersampling(
                    df_dados=df_test_combined,
                    output_csv=None,
                    embaralhar=False
                )
            # Se test_csv for um único arquivo (string)
            else:
                df_test = pd.read_csv(test_csv)
                df_test_reorg = reorganizar_dataset(
                    df_dados=df_test,
                    n_amostras=params['n_amostras'],
                    janelamento=params['janelamento'],
                    amostras_repetidas=params['amostras_repetidas'] if params['janelamento'] else None,
                    salvar_csv=False
                )
                df_test_balanceado = balancear_csv_por_undersampling(
                    df_dados=df_test_reorg,
                    output_csv=None,
                    embaralhar=False
                )
        
        # Balanceia o dataset principal
        df_balanceado = balancear_csv_por_undersampling(
            df_dados=df_combined,
            output_csv=None,
            embaralhar=False
        )
        
        df_reconstruido, df_latente_treino, df_latente_teste, dim_lat, loss_history = processar_autoencoder(
            df_original=df_balanceado,
            params_autoencoder={
                'input_dim': params['n_amostras'],
                'latent_dim': params['latent_dim'],
                'hidden_dim': params['hidden_dim']
            },
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            train_size=params['train_size'],
            df_latente_input=df_test_balanceado,
            return_losses=True
        )

        resultados_balanceado = avaliar_modelos(df_balanceado, df_test_balanceado)
        resultados_latente = avaliar_modelos(df_latente_treino, df_latente_teste)

        melhor_classificador = max(resultados_balanceado.items(), 
                                 key=lambda x: x[1]['Acuracia'])[0]
        melhor_acuracia = resultados_balanceado[melhor_classificador]['Acuracia']

        melhor_classificador_latente = max(resultados_latente.items(),
                                         key=lambda x: x[1]['Acuracia'])[0]
        melhor_acuracia_latente = resultados_latente[melhor_classificador_latente]['Acuracia']

        resultados_top5_balanceado.append({
            'exec_id': exec_id,
            'classificador': melhor_classificador,
            'dataset': 'balanceado',
            'acuracia': melhor_acuracia,
            'parametros': params
        })

        resultados_top5_latente.append({
            'exec_id': exec_id,
            'classificador': melhor_classificador_latente,
            'dataset': 'latente',
            'acuracia': melhor_acuracia_latente,
            'parametros': params
        })

        for classificador, metricas in resultados_balanceado.items():
            if metricas['Acuracia'] >= 0.5 and resultados_latente[classificador]['Acuracia'] > metricas['Acuracia']:
                resultados_latente_melhor.append({
                    'exec_id': exec_id,
                    'classificador': classificador,
                    'acuracia_balanceado': metricas['Acuracia'],
                    'acuracia_latente': resultados_latente[classificador]['Acuracia'],
                    'parametros': params
                })

        metadados['execucoes'][exec_id] = {
            'params': params,
            'resultados_balanceado': resultados_balanceado,
            'resultados_latente': resultados_latente,
            'melhor_classificador': melhor_classificador,
            'melhor_classificador_latente': melhor_classificador_latente,
            'loss_history': loss_history
        }

    resultados_top5_balanceado.sort(key=lambda x: x['acuracia'], reverse=True)
    num_top5 = max(1, int(len(resultados_top5_balanceado) * 0.05))
    top5_final = resultados_top5_balanceado[:num_top5]
    
    resultados_top5_latente.sort(key=lambda x: x['acuracia'], reverse=True)
    num_top5_latente = max(1, int(len(resultados_top5_latente) * 0.05))
    top5_latente_final = resultados_top5_latente[:num_top5_latente]
    
    with open(os.path.join(pasta_top5_balanceado, 'resultados_top5.json'), 'w', encoding='utf-8') as f:
        json.dump(top5_final, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(pasta_top5_latente, 'resultados_top5_latente.json'), 'w', encoding='utf-8') as f:
        json.dump(top5_latente_final, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(pasta_latente_melhor, 'resultados_latente_melhor.json'), 'w', encoding='utf-8') as f:
        json.dump(resultados_latente_melhor, f, indent=2, ensure_ascii=False)
    
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

        input_csvs = [str(DATASETS_PROC / f) for f in [
    "processado_dataset_A1_01_07_NA.csv",
    "processado_dataset_A2_02_10.csv",
    "processado_dataset_A2_08_08.csv",
    "processado_dataset_A2_09_07_NA.csv",
    "processado_dataset_A2_12_08.csv",
    "processado_dataset_A2_14_10.csv",
    "processado_dataset_A2_28_08.csv",
    "processado_dataset_A3_04_12_NA.csv",
    "processado_dataset_A3_09_12.csv",
    "processado_dataset_A3_11_12.csv",
    "processado_dataset_A4_06_01.csv",
    "processado_dataset_A4_13_01.csv",
    "processado_dataset_A4_16_12_NA.csv",
    "processado_dataset_A4_19_12.csv",
    ]],
    test_csv = [str(DATASETS_PROC / f) for f in [
    "processado_dataset_A5_22_01_NA.csv",
    "processado_dataset_A5_27_01.csv",
    "processado_dataset_A5_28_01.csv",
    ]],
        lista_time_ranges=[[(0, 18000, 0), (54000, 500000, 1)]],
        lista_grey_zones=[(18000, 54000)],
        lista_n_amostras=[4,8,12,16,20,24,28,32],
        lista_janelamento=[True],
        lista_amostras_repetidas=[2, 4, 6, 8, 10],
        lista_latent_dims=[1, 2, 4, 6, 8, 10],
        lista_hidden_dims=[8, 16, 32, 64, 128],
        lista_learning_rates=[0.005],
        lista_epochs=[400],
        lista_batch_sizes=[32],
        lista_train_sizes=[0.7],

    )