import json
import matplotlib.pyplot as plt
import os
import numpy as np
from paths import PLOTS_DIR

def plot_accuracy_comparisons(json_path, output_dir=None, metric='Acuracia'):
    if output_dir is None:
        output_dir = str(PLOTS_DIR)
        os.makedirs(output_dir, exist_ok=True)

    # 1. Carregar dados
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Erro ao carregar JSON: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 2. Lista de parâmetros alvo
    TARGET_PARAMS = [
        'n_amostras',
        'amostras_repetidas', 
        'latent_dim',
        'hidden_dim'
    ]

    # 3. Configuração dos gráficos
    # plt.style.use('seaborn')
    colors = {
        'balanceado': '#3498db',  # Azul original
        'latente': '#e74c3c'      # Vermelho original
    }

    # 4. Processamento e plotagem
    print("\n=== Gerando gráficos ===")
    for method in ['regressao_logistica', 'SVM(RBF)', 'arvore_de_decisao']:
        print(f"\nMétodo: {method}")
        
        for param in TARGET_PARAMS:
            # Verificar se o parâmetro existe nos dados
            execucoes_com_param = [e for e in data['execucoes'].values() if param in e['params']]
            if not execucoes_com_param:
                print(f" - {param}: Nenhum dado (pulando)")
                continue

            # Coletar dados separadamente
            dados = {
                'balanceado': {'x': [], 'y': []},
                'latente': {'x': [], 'y': []}
            }

            for exec_data in execucoes_com_param:
                try:
                    param_value = exec_data['params'][param]
                    dados['balanceado']['x'].append(param_value)
                    dados['balanceado']['y'].append(exec_data['resultados_balanceado'][method][metric])
                    dados['latente']['x'].append(param_value)
                    dados['latente']['y'].append(exec_data['resultados_latente'][method][metric])
                except KeyError as e:
                    print(f"   ! Erro em {param}: {e}")
                    continue

            # Criar figura com 3 subplots (2 gráficos + valores)
            fig = plt.figure(figsize=(18, 6))
            gs = fig.add_gridspec(1, 3, width_ratios=[3, 3, 1])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharey=ax1)
            ax3 = fig.add_subplot(gs[2])
            
            fig.suptitle(f'{method} - Variação por {param}', fontsize=14, y=1.02)

            # Plotar dados balanceados
            x_bal = np.array(dados['balanceado']['x'])
            y_bal = np.array(dados['balanceado']['y'])
            sort_idx = np.argsort(x_bal)
            x_bal_sorted = x_bal[sort_idx]
            y_bal_sorted = y_bal[sort_idx]
            
            # Usar valores exatos no eixo X e ajustar espaçamento
            ax1.scatter(x_bal_sorted, y_bal_sorted,
                       color=colors['balanceado'], s=80,
                       alpha=0.8, label='Balanceado')
            
            # Definir ticks exatos para o eixo X
            ax1.set_xticks(np.unique(x_bal_sorted))
            
            # Plotar dados latentes
            x_lat = np.array(dados['latente']['x'])
            y_lat = np.array(dados['latente']['y'])
            sort_idx = np.argsort(x_lat)
            x_lat_sorted = x_lat[sort_idx]
            y_lat_sorted = y_lat[sort_idx]
            
            ax2.scatter(x_lat_sorted, y_lat_sorted,
                       color=colors['latente'], s=80,
                       alpha=0.8, label='Latente')
            
            # Definir ticks exatos para o eixo X
            ax2.set_xticks(np.unique(x_lat_sorted))

            # Configurações comuns para os gráficos
            for ax in (ax1, ax2):
                # Ajustar limites do eixo X com base nos valores reais
                unique_vals = np.unique(ax.get_xticks())
                if len(unique_vals) > 1:
                    padding = (unique_vals[-1] - unique_vals[0]) * 0.1  # Reduzi o padding
                    ax.set_xlim(unique_vals[0] - padding, unique_vals[-1] + padding)
                
                ax.set_ylim(0.0, 1.05)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(loc='lower right')
                ax.set_xlabel(param, fontsize=12)
                ax.set_ylabel(metric, fontsize=12)

            ax1.set_title('Dados Balanceados')
            ax2.set_title('Dados Latentes')

            # Adicionar valores no terceiro subplot
            ax3.axis('off')  # Desativa os eixos
            
            # Calcular estatísticas
            stats_bal = {
                'Média': np.mean(y_bal),
                'Mediana': np.median(y_bal),
                'Desvio': np.std(y_bal),
                'Mínimo': np.min(y_bal),
                'Máximo': np.max(y_bal)
            }
            
            stats_lat = {
                'Média': np.mean(y_lat),
                'Mediana': np.median(y_lat),
                'Desvio': np.std(y_lat),
                'Mínimo': np.min(y_lat),
                'Máximo': np.max(y_lat)
            }
            
            # Criar texto formatado
            stats_text = "Estatísticas de Acurácia:\n\n"
            stats_text += "Balanceado:\n"
            for k, v in stats_bal.items():
                stats_text += f"{k}: {v:.4f}\n"
            
            stats_text += "\nLatente:\n"
            for k, v in stats_lat.items():
                stats_text += f"{k}: {v:.4f}\n"
            
            ax3.text(0.1, 0.5, stats_text, ha='left', va='center', fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

            # Ajustar layout
            plt.tight_layout()

            # Salvamento
            filename = os.path.join(output_dir, f"{method}_{param}_separado.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f" - {param}: Gráfico salvo em {filename}")

    print("\n=== Processo concluído ===")
    print(f"Gráficos salvos em: {os.path.abspath(output_dir)}")

# Exemplo de uso

if __name__ == "__main__":

    caminho_json = r'resultados_personalizados_20250818_162933\metadados_completos.json'

    plot_accuracy_comparisons(caminho_json)