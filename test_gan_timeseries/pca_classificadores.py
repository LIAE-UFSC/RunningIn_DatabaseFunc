import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
from pathlib import Path

# Configuração de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Função para carregar e preparar os dados (atualizada para classes A, B, C)
def carregar_dados(caminho_arquivo):
    caminho_arquivo = Path(caminho_arquivo)
    
    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

    df = pd.read_csv(caminho_arquivo)
    
    # Verificar se a coluna 'anomaly' existe e contém A, B, C
    if 'anomaly' not in df.columns:
        raise ValueError("Coluna 'anomaly' não encontrada no dataset")
    
    # Verificar classes presentes
    classes_presentes = df['anomaly'].unique()
    expected_classes = {'A', 'B', 'C'}
    
    if not set(classes_presentes).issubset(expected_classes):
        raise ValueError(f"O dataset deve conter apenas classes A, B, C. Classes encontradas: {classes_presentes}")
    
    # Codificar as classes para números (A->0, B->1, C->2)
    le = LabelEncoder()
    df['anomaly_encoded'] = le.fit_transform(df['anomaly'])  # Isso converterá A,B,C para 0,1,2
    
    # Extrair features e labels
    X = df['massFlow'].values.astype(np.float32)  # Usando massFlow como feature
    y = df['anomaly_encoded'].values.astype(np.int64)  # Classes codificadas
    
    # Normalização
    X = (X - X.mean()) / X.std()
    
    # Converter para tensores
    X_tensor = torch.tensor(X).unsqueeze(1).to(device)
    y_tensor = torch.tensor(y).to(device)
    
    # Criar dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Dividir em treino (70%), validação (15%) e teste (15%)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    return train_dataset, val_dataset, test_dataset, le

# [O resto do código permanece igual até a função avaliar_modelo_completo...]

# 4. Função de Avaliação Completa (atualizada para mostrar A, B, C)
def avaliar_modelo_completo(model, dataset, le, dataset_name="Teste"):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    
    all_labels = []
    all_preds = []
    all_features = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            outputs, features = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_features.extend(features.cpu().numpy())
    
    # Converter números de volta para A, B, C
    all_labels = le.inverse_transform(all_labels)
    all_preds = le.inverse_transform(all_preds)
    
    # 1. Relatório de Classificação Detalhado
    print(f"\n{'='*50}")
    print(f"AVALIAÇÃO NO CONJUNTO DE {dataset_name.upper()}")
    print(f"{'='*50}")
    
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(
        all_labels, all_preds, 
        target_names=['Classe A', 'Classe B', 'Classe C'],
        digits=4
    ))
    
    # 2. Matriz de Confusão
    cm = confusion_matrix(all_labels, all_preds, labels=['A', 'B', 'C'])
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Classe A', 'Classe B', 'Classe C'],
                yticklabels=['Classe A', 'Classe B', 'Classe C'])
    plt.title(f'Matriz de Confusão - {dataset_name}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.show()
    
    # 3. Métricas por Classe
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=['A', 'B', 'C'])
    print("\nMétricas por Classe:")
    metrics_df = pd.DataFrame({
        'Classe': ['A', 'B', 'C'],
        'Precisão': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    print(metrics_df.to_string(index=False))
    
    return np.array(all_features), all_labels

# 5. Visualizações (atualizada para mostrar A, B, C)
def visualizar_espaco_latente(features, labels, le=None, title="Espaço Latente"):
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Se labels forem numéricas (0,1,2), converter para A,B,C
    if le and all(isinstance(label, (int, np.integer)) for label in labels[:10]):  # Verifica os primeiros 10
        labels = le.inverse_transform(labels)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=[ord(l) for l in labels], cmap='viridis', alpha=0.7)
    
    # Criar colorbar com A, B, C
    classes = sorted(set(labels))
    cbar = plt.colorbar(scatter, ticks=[ord(c) for c in classes])
    cbar.ax.set_yticklabels(classes)
    cbar.set_label('Classes')
    
    plt.title(f'{title} (PCA 2D) - Variância Explicada: {pca.explained_variance_ratio_.sum():.2f}')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True)
    plt.show()

# 6. Pipeline Completo (atualizada)
def pipeline_completa():
    # 1. Carregar dados
    caminho_arquivo = caminho_arquivo = r'C:\Users\PC-1\Documents\GitHub\RunningIn_DatabaseFunc\test_gan_timeseries\dataset_massflow_A1_com_labels.csv'  # Substitua pelo seu caminho
    train_dataset, val_dataset, test_dataset, le = carregar_dados(caminho_arquivo)
    
    print(f"\nDistribuição dos Dados:")
    print(f"- Treino: {len(train_dataset)} amostras")
    print(f"- Validação: {len(val_dataset)} amostras")
    print(f"- Teste: {len(test_dataset)} amostras")
    
    # 2. Treinar modelo
    print("\nIniciando Treinamento...")
    model, historico = treinar_modelo(train_dataset, val_dataset, num_epochs=100)
    
    # 3. Avaliar em todos os conjuntos
    print("\nAvaliando no Conjunto de Treino...")
    train_features, train_labels = avaliar_modelo_completo(model, train_dataset, le, "Treino")
    
    print("\nAvaliando no Conjunto de Validação...")
    val_features, val_labels = avaliar_modelo_completo(model, val_dataset, le, "Validação")
    
    print("\nAvaliando no Conjunto de Teste...")
    test_features, test_labels = avaliar_modelo_completo(model, test_dataset, le, "Teste")
    
    # 4. Visualizações
    plotar_metricas(historico)
    
    print("\nVisualizando Espaço Latente para Treino...")
    visualizar_espaco_latente(train_features, train_labels, le, "Espaço Latente (Treino)")
    
    print("\nVisualizando Espaço Latente para Teste...")
    visualizar_espaco_latente(test_features, test_labels, le, "Espaço Latente (Teste)")
    
    return model, le

if __name__ == "__main__":
    modelo_treinado, label_encoder = pipeline_completa()