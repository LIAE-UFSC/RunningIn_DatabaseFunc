import os
import itertools
import pandas as pd
from datetime import datetime
from autoeoncoderNNpy import BaseModel, Autoencoder, processar_autoencoder, plot_autoencoder_results
from funcao_rotulos import label_dataset_by_time
from funcao_janelamento import reorganizar_dataset
from funcao_random_undersampling import balancear_csv_por_undersampling


def executar_busca_em_grade(
    input_csv='dataset_massflow.csv',
    # Parâmetros fixos para label_dataset_by_time
    time_ranges=[(0, 18000, 0), (54000, 100000, 1)],
    grey_zone=(18000, 54000),
    # Parâmetros variáveis para reorganizar_dataset
    lista_n_amostras=[5, 8, 10],
    lista_janelamento=[True, False],
    lista_amostras_repetidas=[1, 4],
    # Parâmetros variáveis para o autoencoder
    lista_latent_dims=[2, 3, 5],
    # Parâmetros fixos adicionais
    learning_rate=0.02,
    epochs=200,
    batch_size=32,
    train_size=0.75
):
    """
    Executa todo o pipeline com busca em grade e retorna resultados organizados.
    """
    # 1. Criar pasta de resultados (com timestamp para evitar sobrescrita)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta_resultados = f"resultados_{timestamp}"
    os.makedirs(pasta_resultados, exist_ok=True)
    
    resultados = {}

    # 2. Etapa 1: Rotulação Fixa
    print(">>> Etapa 1/4: Rotulando dados temporais...")
    df_labeled, _ = label_dataset_by_time(
        input_csv=input_csv,
        time_ranges=time_ranges,
        grey_zone=grey_zone,
        exclude_grey=True,
        save_greyzone=True,
        greyzone_csv=os.path.join(pasta_resultados, 'greyzone_dataset.csv'),
        output_csv=os.path.join(pasta_resultados, 'dataset_rotulado.csv')
    )

    # 3. Etapa 2: Gerar combinações de parâmetros
    combinacoes_reorg = []
    for n_amostras in lista_n_amostras:
        for janelamento in lista_janelamento:
            if janelamento:
                for amostras_repetidas in lista_amostras_repetidas:
                    combinacoes_reorg.append({
                        'n_amostras': n_amostras,
                        'janelamento': janelamento,
                        'amostras_repetidas': amostras_repetidas
                    })
            else:
                combinacoes_reorg.append({
                    'n_amostras': n_amostras,
                    'janelamento': janelamento,
                    'amostras_repetidas': None  # Não aplicável
                })

    # 4. Processar cada combinação
    print(f"\n>>> Etapa 2/4: Reorganizando dados ({len(combinacoes_reorg)} combinações)...")
    for i, combo in enumerate(combinacoes_reorg):
        print(f"\nCombinação {i+1}: n_amostras={combo['n_amostras']}, janelamento={combo['janelamento']}, repeticoes={combo['amostras_repetidas']}")

        # Reorganização
        df_reorg = reorganizar_dataset(
            caminho_arquivo=os.path.join(pasta_resultados, 'dataset_rotulado.csv'),
            n_amostras=combo['n_amostras'],
            incluir_tempo=False,
            rotulo_ultimo=True,
            salvar_csv=True,
            nome_saida=os.path.join(pasta_resultados, f'dataset_reorg_{i+1}.csv'),
            janelamento=combo['janelamento'],
            amostras_repetidas=combo['amostras_repetidas'] if combo['janelamento'] else 1
        )

        # Balanceamento
        print(">>> Etapa 3/4: Balanceando dados...")
        df_balanceado = balancear_csv_por_undersampling(
            input_csv=os.path.join(pasta_resultados, f'dataset_reorg_{i+1}.csv'),
            output_csv=os.path.join(pasta_resultados, f'dataset_balanceado_{i+1}.csv'),
            embaralhar=False
        )

        # 5. Autoencoder para cada dimensão latente
        print(">>> Etapa 4/4: Processando autoencoder...")
        for latent_dim in lista_latent_dims:
            print(f"  - latent_dim={latent_dim}")
            params_ae = {
                'input_dim': combo['n_amostras'],
                'latent_dim': latent_dim
            }

            df_reconstruido, df_latente, _ = processar_autoencoder(
                df_original=df_balanceado,
                params_autoencoder=params_ae,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                train_size=train_size
            )

            # Salvar resultados
            suffix = f"comb{i+1}_latent{latent_dim}"
            df_reconstruido.to_csv(
                os.path.join(pasta_resultados, f'reconstruido_{suffix}.csv'), 
                index=False
            )
            df_latente.to_csv(
                os.path.join(pasta_resultados, f'latente_{suffix}.csv'), 
                index=False
            )

            # Registrar metadados
            resultados[suffix] = {
                'params_preprocess': combo,
                'params_autoencoder': params_ae,
                'caminhos': {
                    'reconstruido': os.path.join(pasta_resultados, f'reconstruido_{suffix}.csv'),
                    'latente': os.path.join(pasta_resultados, f'latente_{suffix}.csv'),
                    'balanceado': os.path.join(pasta_resultados, f'dataset_balanceado_{i+1}.csv')
                }
            }

    # 6. Salvar sumário executivo
    df_sumario = pd.DataFrame.from_dict(resultados, orient='index')
    df_sumario.to_csv(os.path.join(pasta_resultados, 'sumario_executivo.csv'), index=True)
    print(f"\n✔ Busca concluída! Resultados salvos em: {pasta_resultados}")

    return {
        'resultados': resultados,
        'pasta_resultados': pasta_resultados,
        'sumario': df_sumario
    }


# Exemplo mínimo
resultados = executar_busca_em_grade(
    lista_n_amostras=[5, 8],
    lista_janelamento=[True, False],
    lista_latent_dims=[2, 3]
)

# Acessar resultados
print(resultados['pasta_resultados'])  # Caminho da pasta
print(resultados['resultados']['comb1_latent2']['params_preprocess'])  # Parâmetros usados