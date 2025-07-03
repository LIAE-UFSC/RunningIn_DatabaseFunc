import json
import matplotlib.pyplot as plt
import numpy as np

def plot_acuracia_medias(metadados):
    """Calcula e plota as médias de acurácia por método e dataset."""
    # Calcula as médias (usando a função anterior)
    medias = calcular_media_acuracia(metadados)
    
    # Preparação dos dados para plotagem
    metodos = list(medias["balanceado"].keys())
    balanceado_values = [medias["balanceado"][m] for m in metodos]
    latente_values = [medias["latente"][m] for m in metodos]
    
    # Configuração do gráfico
    x = np.arange(len(metodos))  # posições dos métodos
    width = 0.35  # largura das barras
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Barras para dataset balanceado
    rects1 = ax.bar(x - width/2, balanceado_values, width, 
                   label='Balanceado', color='#1f77b4')
    
    # Barras para dataset latente
    rects2 = ax.bar(x + width/2, latente_values, width, 
                   label='Latente', color='#ff7f0e')
    
    # Adiciona texto, rótulos e título
    ax.set_ylabel('Acurácia Média')
    ax.set_title('Acurácia Média por Método e Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(metodos)
    ax.legend()
    
    # Adiciona valores nas barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    # Ajusta layout e mostra o gráfico
    fig.tight_layout()
    plt.ylim(0, 1)  # Acurácia vai de 0 a 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Função para calcular médias (igual à anterior)
def calcular_media_acuracia(metadados):
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

# Plotagem
plot_acuracia_medias(metadados)

