"""Generation (legacy) of metric heatmaps by n_amostras × amostras_repetidas.

Exploratory script that reads a ``metadados_completos.json`` from the grid search and
saves one heatmap per method (balanced data). Not imported by the current pipeline.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
from paths import HEATMAPS_DIR

def plot_balanceado_heatmaps(json_path, output_dir=None, metric='Acuracia'):
    """Generate 3 heatmaps (one per method) of the metric for balanced data,
    with n_amostras on the X axis and amostras_repetidas on the Y axis."""
    if output_dir is None:
        output_dir = str(HEATMAPS_DIR)
        os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 2. Extract all unique values found in the data for the axes
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

    # 3. Classification methods
    metodos = ['regressao_logistica', 'SVM(RBF)', 'arvore_de_decisao']
    metodo_names = {
        'regressao_logistica': 'Logistic Regression',
        'SVM(RBF)': 'SVM (RBF)',
        'arvore_de_decisao': 'Decision Tree'
    }

    # 4. Heatmap configuration
    cmap = LinearSegmentedColormap.from_list('custom', ['#e74c3c', '#f1c40f', '#2ecc71'])
    plt.style.use('seaborn-v0_8')

    # 5. Generate a heatmap for each method
    for metodo in metodos:
        # Create a matrix of NaNs
        heatmap_data = np.full((len(amostras_repetidas), len(n_amostras)), np.nan)

        # Map indices
        x_index = {val: idx for idx, val in enumerate(n_amostras)}
        y_index = {val: idx for idx, val in enumerate(amostras_repetidas)}

        # Fill the matrix with the values
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
                        heatmap_data[y, x] = valor  # fill even if it is zero
                    except KeyError:
                        print(f"Warning: no data for {metodo}, metric '{metric}' at (n={x_val}, r={y_val})")

        # Mask for the values that remain NaN (not computed)
        mask = np.isnan(heatmap_data)

        # Create the heatmap
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

        # Visual settings
        plt.title(f'{metodo_names[metodo]} - {metric} (Balanced)', pad=20)
        plt.xlabel('Number of samples', labelpad=10)
        plt.ylabel('Repeated samples', labelpad=10)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save the plot
        filename = os.path.join(output_dir, f'heatmap_{metodo}_{metric}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved: {filename}")

    print(f"\nAll heatmaps saved to: {os.path.abspath(output_dir)}")

# Usage example

caminho_json = r'resultados_personalizados_20250818_162933\metadados_completos.json'

if __name__ == "__main__":
    plot_balanceado_heatmaps(
        caminho_json,
        metric='F1_score'
    )
