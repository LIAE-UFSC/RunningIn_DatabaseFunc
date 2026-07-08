"""Analysis (legacy) of the cases where the latent space beats the balanced one.

Exploratory script run by hand on a ``resultados_latente_melhor.json``.
Not imported by the current pipeline.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

def analise_simples(json_path):
    """Read the results JSON and plot the accuracy difference (latent − balanced)
    by classifier and by latent dimension, for the cases where the latent was better."""
    # Load the data
    with open(json_path, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    df = pd.DataFrame(dados)

    # Compute the accuracy difference
    df['diferenca'] = df['acuracia_latente'] - df['acuracia_balanceado']

    # Filter cases where the latent was better
    df_melhor = df[df['diferenca'] > 0]

    if df_melhor.empty:
        print("No case where the latent method beat the balanced one")
        return

    # Extract important parameters
    df_melhor['latent_dim'] = df_melhor['parametros'].apply(lambda x: x['latent_dim'])

    # Visual configuration
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 5))

    # Plot 1: mean difference by classifier
    plt.subplot(1, 2, 1)
    df_media = df_melhor.groupby('classificador')['diferenca'].mean().reset_index()
    sns.barplot(x='classificador', y='diferenca', data=df_media)
    plt.title('Mean difference by classifier')
    plt.xticks(rotation=45)
    plt.ylabel('Difference (Latent - Balanced)')

    # Plot 2: relationship between latent dimension and difference
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='latent_dim', y='diferenca', data=df_melhor)
    plt.title('Difference by latent dimension')
    plt.xlabel('Latent dimension')
    plt.ylabel('Difference')

    plt.tight_layout()
    plt.show()

# File path and function call
json_path = r'C:\Users\pedro\OneDrive\Área de Trabalho\geral\GitHubAll\RunningIn_DatabaseFunc\classificator_project\resultados_personalizados_20250703_120333\latente_melhor\resultados_latente_melhor.json'
analise_simples(json_path)
