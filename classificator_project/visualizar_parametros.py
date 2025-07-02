import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_accuracy_comparisons(json_path, output_dir='plots', metric='Acuracia'):
    # Carregar dados
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuração de estilo minimalista
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 0.5,
    })
    
    # Cores e marcadores
    colors = {
        'balanceado': '#3498db',  # Azul
        'latente': '#e74c3c'      # Vermelho
    }
    
    # Processar dados para cada método
    for method in ['regressao_logistica', 'SVM(RBF)', 'arvore_de_decisao']:
        for param in ['n_amostras', 'amostras_repetidas', 'latent_dim', 'janelamento']:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Coletar dados
            x_vals = []
            y_bal = []
            y_lat = []
            
            for exec_name, exec_data in data['execucoes'].items():
                if param in exec_data['params']:
                    x_vals.append(exec_data['params'][param])
                    y_bal.append(exec_data['resultados_balanceado'][method][metric])
                    y_lat.append(exec_data['resultados_latente'][method][metric])
            
            # Ordenar por valor do parâmetro
            sort_idx = np.argsort(x_vals)
            x_sorted = np.array(x_vals)[sort_idx]
            y_bal_sorted = np.array(y_bal)[sort_idx]
            y_lat_sorted = np.array(y_lat)[sort_idx]
            
            # Plotar pontos (exatamente como antes)
            ax.scatter(x_sorted, y_bal_sorted, 
                      color=colors['balanceado'],
                      s=60, alpha=0.8,
                      label='Balanceado')
            
            ax.scatter(x_sorted, y_lat_sorted,
                      color=colors['latente'],
                      s=60, alpha=0.8,
                      label='Latente')
            
            # Configurações mínimas
            ax.set_title(f'{method} - {param}', 
                        fontsize=12, pad=10)
            ax.set_xlabel(param, fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.set_ylim(0.0, 1.05)  # Ajustado para todas as métricas
            
            # Legenda simples
            ax.legend(frameon=False, fontsize=9)
            
            # Remover grid
            ax.grid(False)
            
            # Salvar exatamente como antes (só muda a métrica no eixo Y)
            filename = f"{output_dir}/{method}_{param}.png"  # Mantém o mesmo nome de arquivo
            plt.savefig(filename, dpi=120, bbox_inches='tight')
            plt.close()
    
    print(f"Gráficos salvos em: {os.path.abspath(output_dir)}")

# Exemplo de uso (igual ao anterior)
metadados_completos = r'classificator_project\resultados_personalizados_20250702_120344\metadados_completos.json'

#plot_accuracy_comparisons(metadados_completos)  # Padrão: Acuracia
plot_accuracy_comparisons(metadados_completos, metric='F1_score')  # Para F1-score

