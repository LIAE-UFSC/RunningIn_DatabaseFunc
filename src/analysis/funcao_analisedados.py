"""Analysis (legacy) of balanced × latent accuracy from the grid search metadata.

Exploratory script run by hand on a ``metadados_completos.json`` produced by
``gridsearch.py``. Not imported by the current pipeline.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

caminho_json = r'resultados_personalizados_20250818_162933\metadados_completos.json'

def carregar_metadados(caminho_arquivo):
    """Load the complete metadata from a JSON file."""
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        return json.load(f)

def analisar_acuracia_comparativa(metadados_completos):
    """
    Analyze and compare the classifiers' accuracy on the balanced datasets
    and on the latent space for each classification method.
    """
    metodos_classificacao = ["regressao_logistica", "SVM(RBF)", "arvore_de_decisao"]
    resultados_comparativos = {metodo: {"melhor_latente": 0, "total_ensaios": 0} for metodo in metodos_classificacao}

    for execucao in metadados_completos["execucoes"].values():
        for metodo in metodos_classificacao:
            acuracia_balanceado = execucao["resultados_balanceado"][metodo]["Acuracia"]
            acuracia_latente = execucao["resultados_latente"][metodo]["Acuracia"]

            resultados_comparativos[metodo]["total_ensaios"] += 1
            if acuracia_latente > acuracia_balanceado:
                resultados_comparativos[metodo]["melhor_latente"] += 1

    percentagens = {}
    for metodo, dados in resultados_comparativos.items():
        if dados["total_ensaios"] > 0:
            percentagem = (dados["melhor_latente"] / dados["total_ensaios"]) * 100
        else:
            percentagem = 0
        percentagens[metodo] = percentagem

    return percentagens

def plot_acuracia_crescente(metadados_completos):
    """Plot the accuracy sorted (increasing) by method and dataset type (balanced/latent)."""
    metodos = ["regressao_logistica", "SVM(RBF)", "arvore_de_decisao"]
    tipos_dataset = ["balanceado", "latente"]

    # Distinct colors for each method-type combination
    cores = {
        "regressao_logistica_balanceado": "#1f77b4",  # blue
        "regressao_logistica_latente": "#ff7f0e",     # orange
        "SVM(RBF)_balanceado": "#2ca02c",            # green
        "SVM(RBF)_latente": "#d62728",               # red
        "arvore_de_decisao_balanceado": "#9467bd",   # purple
        "arvore_de_decisao_latente": "#8c564b"       # brown
    }

    # Prepare the data
    dados = {metodo: {tipo: [] for tipo in tipos_dataset} for metodo in metodos}

    for execucao in metadados_completos["execucoes"].values():
        for metodo in metodos:
            for tipo in tipos_dataset:
                acuracia = execucao[f"resultados_{tipo}"][metodo]["Acuracia"]
                dados[metodo][tipo].append(acuracia)

    # Sort the data (increasing) and plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for metodo in metodos:
        for tipo in tipos_dataset:
            chave_cor = f"{metodo}_{tipo}"
            acuracias_ordenadas = sorted(dados[metodo][tipo])
            x = range(1, len(acuracias_ordenadas) + 1)
            ax.plot(
                x,
                acuracias_ordenadas,
                label=f"{metodo} ({tipo})",
                color=cores[chave_cor],
                linestyle='--' if tipo == 'balanceado' else '-',
                marker='o' if metodo == 'regressao_logistica' else ('s' if metodo == 'SVM(RBF)' else '^')
            )

    ax.set_title("Increasing accuracy by method and dataset type")
    ax.set_xlabel("Ordered runs (lowest to highest accuracy)")
    ax.set_ylabel("Accuracy")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

try:
    metadados_completos = carregar_metadados(caminho_json)

    percentagens_melhor_latente = analisar_acuracia_comparativa(metadados_completos)

    for metodo, percentagem in percentagens_melhor_latente.items():
        print(f"The latent dataset had better accuracy in {percentagem:.2f}% of the runs for the '{metodo}' method.")

    plot_acuracia_crescente(metadados_completos)

except FileNotFoundError:
    print(f"Error: file not found at path {caminho_json}")
except json.JSONDecodeError:
    print(f"Error: file {caminho_json} is not valid JSON")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
