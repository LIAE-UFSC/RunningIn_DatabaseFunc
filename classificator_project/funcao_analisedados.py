import json
import matplotlib.pyplot as plt
import numpy as np

caminho_json = r'resultados_personalizados_20250703_100856\metadados_completos.json'

def carregar_metadados(caminho_arquivo):
    """Carrega os metadados completos de um arquivo JSON."""
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        return json.load(f)

def analisar_acuracia_comparativa(metadados_completos):
    """
    Analisa e compara a acurácia dos classificadores em datasets balanceados
    e no espaço latente para cada método de classificação.
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
    metodos = ["regressao_logistica", "SVM(RBF)", "arvore_de_decisao"]
    tipos_dataset = ["balanceado", "latente"]
    
    # Cores distintas para cada combinação método-tipo
    cores = {
        "regressao_logistica_balanceado": "#1f77b4",  # azul
        "regressao_logistica_latente": "#ff7f0e",     # laranja
        "SVM(RBF)_balanceado": "#2ca02c",            # verde
        "SVM(RBF)_latente": "#d62728",               # vermelho
        "arvore_de_decisao_balanceado": "#9467bd",   # roxo
        "arvore_de_decisao_latente": "#8c564b"       # marrom
    }
    
    # Preparar os dados
    dados = {metodo: {tipo: [] for tipo in tipos_dataset} for metodo in metodos}
    
    for execucao in metadados_completos["execucoes"].values():
        for metodo in metodos:
            for tipo in tipos_dataset:
                acuracia = execucao[f"resultados_{tipo}"][metodo]["Acuracia"]
                dados[metodo][tipo].append(acuracia)
    
    # Ordenar os dados (crescente) e plotar
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
    
    ax.set_title("Acurácia Crescente por Método e Tipo de Dataset")
    ax.set_xlabel("Execuções Ordenadas (Menor para Maior Acurácia)")
    ax.set_ylabel("Acurácia")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

try:
    metadados_completos = carregar_metadados(caminho_json)
    
    percentagens_melhor_latente = analisar_acuracia_comparativa(metadados_completos)

    for metodo, percentagem in percentagens_melhor_latente.items():
        print(f"O dataset latente teve melhor acurácia em {percentagem:.2f}% dos ensaios para o método '{metodo}'.")

    plot_acuracia_crescente(metadados_completos)

except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado no caminho {caminho_json}")
except json.JSONDecodeError:
    print(f"Erro: O arquivo {caminho_json} não é um JSON válido")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {str(e)}")