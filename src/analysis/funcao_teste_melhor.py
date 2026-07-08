"""Analysis (legacy) of the mean accuracy by method and dataset type.

Exploratory script run by hand on a ``metadados_completos.json`` from the grid search.
Not imported by the current pipeline.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_acuracia_medias(metadados):
    """Compute and plot the mean accuracy by method and dataset."""
    # Compute the means (using the function below)
    medias = calcular_media_acuracia(metadados)

    # Prepare the data for plotting
    metodos = list(medias["balanceado"].keys())
    balanceado_values = [medias["balanceado"][m] for m in metodos]
    latente_values = [medias["latente"][m] for m in metodos]

    # Plot configuration
    x = np.arange(len(metodos))  # method positions
    width = 0.35  # bar width

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bars for the balanced dataset
    rects1 = ax.bar(x - width/2, balanceado_values, width,
                   label='Balanced', color='#1f77b4')

    # Bars for the latent dataset
    rects2 = ax.bar(x + width/2, latente_values, width,
                   label='Latent', color='#ff7f0e')

    # Add text, labels and title
    ax.set_ylabel('Mean accuracy')
    ax.set_title('Mean accuracy by method and dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(metodos)
    ax.legend()

    # Add values on the bars
    def autolabel(rects):
        """Annotate the value (height) above each bar."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    # Adjust layout and show the plot
    fig.tight_layout()
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Function to compute the means (same as above)
def calcular_media_acuracia(metadados):
    """Compute the mean accuracy by method, for the balanced and latent datasets."""
    balanceado = {
        "regressao_logistica": {"soma": 0, "contagem": 0},
        "SVM(RBF)": {"soma": 0, "contagem": 0},
        "arvore_de_decisao": {"soma": 0, "contagem": 0}
    }
    latente = {
        "regressao_logistica": {"soma": 0, "contagem": 0},
        "SVM(RBF)": {"soma": 0, "contagem": 0},
        "arvore_de_decisao": {"soma": 0, "contagem": 0}
    }

    for execucao in metadados["execucoes"].values():
        for metodo, resultados in execucao["resultados_balanceado"].items():
            balanceado[metodo]["soma"] += resultados["Acuracia"]
            balanceado[metodo]["contagem"] += 1

        for metodo, resultados in execucao["resultados_latente"].items():
            latente[metodo]["soma"] += resultados["Acuracia"]
            latente[metodo]["contagem"] += 1

    return {
        "balanceado": {metodo: dados["soma"]/dados["contagem"] for metodo, dados in balanceado.items()},
        "latente": {metodo: dados["soma"]/dados["contagem"] for metodo, dados in latente.items()}
    }


with open(r'resultados_personalizados_20250703_143321\metadados_completos.json', 'r') as f:
    metadados = json.load(f)

# Plot
plot_acuracia_medias(metadados)
