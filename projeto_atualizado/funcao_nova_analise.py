import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

def analise_simples(json_path):
    # Carrega os dados
    with open(json_path, 'r', encoding='utf-8') as f:
        dados = json.load(f)
    
    df = pd.DataFrame(dados)
    
    # Calcula diferença de acurácia
    df['diferenca'] = df['acuracia_latente'] - df['acuracia_balanceado']
    
    # Filtra casos onde latente foi melhor
    df_melhor = df[df['diferenca'] > 0]
    
    if df_melhor.empty:
        print("Nenhum caso onde o método latente foi melhor que o balanceado")
        return
    
    # Extrai parâmetros importantes
    df_melhor['latent_dim'] = df_melhor['parametros'].apply(lambda x: x['latent_dim'])
    
    # Configuração visual
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 5))
    
    # Gráfico 1: Média da diferença por classificador
    plt.subplot(1, 2, 1)
    df_media = df_melhor.groupby('classificador')['diferenca'].mean().reset_index()
    sns.barplot(x='classificador', y='diferenca', data=df_media)
    plt.title('Média da Diferença por Classificador')
    plt.xticks(rotation=45)
    plt.ylabel('Diferença (Latente - Balanceado)')
    
    # Gráfico 2: Relação entre dimensão latente e diferença
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='latent_dim', y='diferenca', data=df_melhor)
    plt.title('Diferença por Dimensão Latente')
    plt.xlabel('Dimensão Latente')
    plt.ylabel('Diferença')
    
    plt.tight_layout()
    plt.show()

# Caminho do arquivo e chamada da função
json_path = r'C:\Users\pedro\OneDrive\Área de Trabalho\geral\GitHubAll\RunningIn_DatabaseFunc\classificator_project\resultados_personalizados_20250703_120333\latente_melhor\resultados_latente_melhor.json'
analise_simples(json_path)