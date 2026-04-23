import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
from paths import HEATMAPS_DIR

def plot_balanceado_heatmaps(json_path, output_dir=None, metric='Acuracia'):
    if output_dir is None:
        output_dir = str(HEATMAPS_DIR)
        os.makedirs(output_dir, exist_ok=True)
    """
    Gera 3 heatmaps (um para cada método) mostrando a métrica para dados balanceados,
    com n_amostras no eixo X e amostras_repetidas no eixo Y.
    """
    
    # 1. Carregar dados
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Erro ao carregar JSON: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 2. Extrair todos os valores únicos encontrados nos dados para os eixos
    n_amostras = sorted(set(
        exec_data['params']['n_amostras']
        for exec_data in data['execucoes'].values()
        if 'n_amostras' in exec_data['params']
    ))
    
    amostras_repetidas = sorted(set(
        exec_data['params']['amostras_repetidas']
        for exec_data in data['execucoes'].values()
        if 'amostras_repetidas' in exec_data['params']
    ))

    # 3. Métodos de classificação
    metodos = ['regressao_logistica', 'SVM(RBF)', 'arvore_de_decisao']
    metodo_names = {
        'regressao_logistica': 'Regressão Logística',
        'SVM(RBF)': 'SVM (RBF)',
        'arvore_de_decisao': 'Árvore de Decisão'
    }

    # 4. Configuração do heatmap
    cmap = LinearSegmentedColormap.from_list('custom', ['#e74c3c', '#f1c40f', '#2ecc71'])
    plt.style.use('seaborn-v0_8')

    # 5. Gerar heatmap para cada método
    for metodo in metodos:
        # Criar matriz de NaNs
        heatmap_data = np.full((len(amostras_repetidas), len(n_amostras)), np.nan)
        
        # Mapear índices
        x_index = {val: idx for idx, val in enumerate(n_amostras)}
        y_index = {val: idx for idx, val in enumerate(amostras_repetidas)}
        
        # Preencher matriz com os valores
        for exec_data in data['execucoes'].values():
            params = exec_data.get('params', {})
            resultados = exec_data.get('resultados_balanceado', {})

            if 'n_amostras' in params and 'amostras_repetidas' in params:
                x_val = params['n_amostras']
                y_val = params['amostras_repetidas']
                if x_val in x_index and y_val in y_index:
                    x = x_index[x_val]
                    y = y_index[y_val]
                    try:
                        valor = resultados[metodo][metric]
                        heatmap_data[y, x] = valor  # preenche mesmo que seja zero
                    except KeyError:
                        print(f"Aviso: Sem dados para {metodo}, métrica '{metric}' em (n={x_val}, r={y_val})")

        # Máscara para os valores que continuam NaN (não computados)
        mask = np.isnan(heatmap_data)

        # Criar o heatmap
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            xticklabels=n_amostras,
            yticklabels=amostras_repetidas,
            cbar_kws={'label': metric},
            mask=mask,
            annot_kws={"fontsize": 9}
        )
        
        # Configurações visuais
        plt.title(f'{metodo_names[metodo]} - {metric} (Balanceado)', pad=20)
        plt.xlabel('Número de Amostras', labelpad=10)
        plt.ylabel('Amostras Repetidas', labelpad=10)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Salvar o gráfico
        filename = os.path.join(output_dir, f'heatmap_{metodo}_{metric}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap salvo: {filename}")

    print(f"\nTodos os heatmaps foram salvos em: {os.path.abspath(output_dir)}")

# Exemplo de uso

caminho_json = r'resultados_personalizados_20250818_162933\metadados_completos.json'

if __name__ == "__main__":
    plot_balanceado_heatmaps(
        caminho_json,
        metric='F1_score'
    )
