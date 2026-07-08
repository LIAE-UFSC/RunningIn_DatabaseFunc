"""Visualization (legacy) of the metric by hyperparameter (balanced × latent).

Exploratory script that reads a ``metadados_completos.json`` from the grid search and
saves, per method and per hyperparameter, plots comparing balanced and latent. Not
imported by the current pipeline.
"""

import json
import matplotlib.pyplot as plt
import os
import numpy as np
from paths import PLOTS_DIR

def plot_accuracy_comparisons(json_path, output_dir=None, metric='Acuracia'):
    """Generate and save, for each method and target hyperparameter, plots of the metric
    (balanced × latent) with summary statistics, in ``output_dir``."""
    if output_dir is None:
        output_dir = str(PLOTS_DIR)
        os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 2. List of target parameters
    TARGET_PARAMS = [
        'n_amostras',
        'amostras_repetidas',
        'latent_dim',
        'hidden_dim'
    ]

    # 3. Plot configuration
    # plt.style.use('seaborn')
    colors = {
        'balanceado': '#3498db',  # Original blue
        'latente': '#e74c3c'      # Original red
    }

    # 4. Processing and plotting
    print("\n=== Generating plots ===")
    for method in ['regressao_logistica', 'SVM(RBF)', 'arvore_de_decisao']:
        print(f"\nMethod: {method}")

        for param in TARGET_PARAMS:
            # Check whether the parameter exists in the data
            execucoes_com_param = [e for e in data['execucoes'].values() if param in e['params']]
            if not execucoes_com_param:
                print(f" - {param}: No data (skipping)")
                continue

            # Collect data separately
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
                    print(f"   ! Error in {param}: {e}")
                    continue

            # Create a figure with 3 subplots (2 plots + values)
            fig = plt.figure(figsize=(18, 6))
            gs = fig.add_gridspec(1, 3, width_ratios=[3, 3, 1])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharey=ax1)
            ax3 = fig.add_subplot(gs[2])

            fig.suptitle(f'{method} - Variation by {param}', fontsize=14, y=1.02)

            # Plot balanced data
            x_bal = np.array(dados['balanceado']['x'])
            y_bal = np.array(dados['balanceado']['y'])
            sort_idx = np.argsort(x_bal)
            x_bal_sorted = x_bal[sort_idx]
            y_bal_sorted = y_bal[sort_idx]

            # Use exact values on the X axis and adjust spacing
            ax1.scatter(x_bal_sorted, y_bal_sorted,
                       color=colors['balanceado'], s=80,
                       alpha=0.8, label='Balanced')

            # Set exact ticks for the X axis
            ax1.set_xticks(np.unique(x_bal_sorted))

            # Plot latent data
            x_lat = np.array(dados['latente']['x'])
            y_lat = np.array(dados['latente']['y'])
            sort_idx = np.argsort(x_lat)
            x_lat_sorted = x_lat[sort_idx]
            y_lat_sorted = y_lat[sort_idx]

            ax2.scatter(x_lat_sorted, y_lat_sorted,
                       color=colors['latente'], s=80,
                       alpha=0.8, label='Latent')

            # Set exact ticks for the X axis
            ax2.set_xticks(np.unique(x_lat_sorted))

            # Common settings for the plots
            for ax in (ax1, ax2):
                # Adjust X-axis limits based on the actual values
                unique_vals = np.unique(ax.get_xticks())
                if len(unique_vals) > 1:
                    padding = (unique_vals[-1] - unique_vals[0]) * 0.1  # Reduced the padding
                    ax.set_xlim(unique_vals[0] - padding, unique_vals[-1] + padding)

                ax.set_ylim(0.0, 1.05)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(loc='lower right')
                ax.set_xlabel(param, fontsize=12)
                ax.set_ylabel(metric, fontsize=12)

            ax1.set_title('Balanced data')
            ax2.set_title('Latent data')

            # Add values in the third subplot
            ax3.axis('off')  # Turn off the axes

            # Compute statistics
            stats_bal = {
                'Mean': np.mean(y_bal),
                'Median': np.median(y_bal),
                'Std': np.std(y_bal),
                'Min': np.min(y_bal),
                'Max': np.max(y_bal)
            }

            stats_lat = {
                'Mean': np.mean(y_lat),
                'Median': np.median(y_lat),
                'Std': np.std(y_lat),
                'Min': np.min(y_lat),
                'Max': np.max(y_lat)
            }

            # Build formatted text
            stats_text = "Accuracy statistics:\n\n"
            stats_text += "Balanced:\n"
            for k, v in stats_bal.items():
                stats_text += f"{k}: {v:.4f}\n"

            stats_text += "\nLatent:\n"
            for k, v in stats_lat.items():
                stats_text += f"{k}: {v:.4f}\n"

            ax3.text(0.1, 0.5, stats_text, ha='left', va='center', fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

            # Adjust layout
            plt.tight_layout()

            # Saving
            filename = os.path.join(output_dir, f"{method}_{param}_separado.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f" - {param}: Plot saved to {filename}")

    print("\n=== Process complete ===")
    print(f"Plots saved to: {os.path.abspath(output_dir)}")

# Usage example

if __name__ == "__main__":

    caminho_json = r'resultados_personalizados_20250818_162933\metadados_completos.json'

    plot_accuracy_comparisons(caminho_json)
