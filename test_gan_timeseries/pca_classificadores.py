import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Configuração de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função para carregar e dividir os dados
def carregar_dados(caminho_arquivo):
    caminho_arquivo = Path(caminho_arquivo)
    
    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

    df = pd.read_csv(caminho_arquivo)
    tempo = df.iloc[:, 0].values.astype(np.float32)  # Coluna time
    dados = df.iloc[:, 1].values.astype(np.float32)   # Coluna massFlow
    rotulos = df.iloc[:, 4].values.astype(np.int64)   # Coluna anomaly (1, 2 ou 3)
    
    # Verificar se todas as 3 classes estão presentes
    classes_presentes = np.unique(rotulos)
    if len(classes_presentes) != 3:
        raise ValueError(f"O dataset deve conter exatamente 3 classes. Classes encontradas: {classes_presentes}")
    
    # Normalização dos dados
    dados = (dados - dados.mean()) / dados.std()
    
    # Converter para tensores PyTorch
    tempo = torch.tensor(tempo).unsqueeze(1).to(device)
    dados = torch.tensor(dados).unsqueeze(1).to(device)
    rotulos = torch.tensor(rotulos).to(device) - 1  # Convertendo classes para 0, 1, 2
    
    # Criar dataset combinado
    dataset = TensorDataset(tempo, dados, rotulos)
    
    # Dividir em treino (75%) e validação (25%)
    tamanho_treino = int(0.75 * len(dataset))
    tamanho_val = len(dataset) - tamanho_treino
    treino_dataset, val_dataset = random_split(dataset, [tamanho_treino, tamanho_val])
    
    return treino_dataset, val_dataset

# Arquitetura da Rede Neural para 3 classes
class ClassificadorSeriesTemporais(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super(ClassificadorSeriesTemporais, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//4, 3)  # 3 classes de saída
        )
        
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits, features

# Função de treinamento
def treinar_modelo(treino_dataset, val_dataset, num_epochs=200, batch_size=32):
    # Criar DataLoaders
    treino_loader = DataLoader(treino_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Inicializar modelo
    model = ClassificadorSeriesTemporais().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # Armazenar métricas
    historico = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Treino
        model.train()
        train_loss, train_correct = 0.0, 0
        for tempo, dados, rotulos in treino_loader:
            optimizer.zero_grad()
            outputs, _ = model(dados)
            loss = criterion(outputs, rotulos)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == rotulos).sum().item()
        
        # Validação
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for tempo, dados, rotulos in val_loader:
                outputs, _ = model(dados)
                loss = criterion(outputs, rotulos)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == rotulos).sum().item()
        
        # Calcular métricas
        train_loss /= len(treino_loader)
        train_acc = train_correct / len(treino_dataset)
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_dataset)
        
        # Atualizar histórico
        historico['train_loss'].append(train_loss)
        historico['val_loss'].append(val_loss)
        historico['train_acc'].append(train_acc)
        historico['val_acc'].append(val_acc)
        
        # Ajustar learning rate
        scheduler.step(val_loss)
        
        # Log
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    return model, historico

# Função para avaliação
def avaliar_modelo(model, dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    
    all_labels = []
    all_preds = []
    all_features = []
    
    with torch.no_grad():
        for tempo, dados, rotulos in loader:
            outputs, features = model(dados)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(rotulos.cpu().numpy() + 1)  # Convertendo de volta para 1, 2, 3
            all_preds.extend(predicted.cpu().numpy() + 1)
            all_features.extend(features.cpu().numpy())
    
    # Métricas de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(all_labels, all_preds, target_names=['Classe 1', 'Classe 2', 'Classe 3']))
    
    # Matriz de confusão
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Classe 1', 'Classe 2', 'Classe 3'], 
                yticklabels=['Classe 1', 'Classe 2', 'Classe 3'])
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.show()
    
    return np.array(all_features), np.array(all_labels)

# Função para visualização do espaço latente
def visualizar_espaco_latente(features, labels):
    # Redução para 2D com PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, ticks=[1, 2, 3], label='Classes')
    plt.title('Espaço Latente (PCA 2D)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True)
    plt.show()

# Função para plotar curvas de aprendizado
def plotar_curvas(historico):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(historico['train_loss'], label='Treino')
    plt.plot(historico['val_loss'], label='Validação')
    plt.title('Curva de Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(historico['train_acc'], label='Treino')
    plt.plot(historico['val_acc'], label='Validação')
    plt.title('Curva de Acurácia')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Pipeline completo
def pipeline_classificacao():
    # 1. Carregar e preparar dados
    caminho_arquivo = r'C:\Users\PC-1\Documents\GitHub\RunningIn_DatabaseFunc\test_gan_timeseries\meu_arquivo_massflow_A1_teste.csv'
    treino_dataset, val_dataset = carregar_dados(caminho_arquivo)
    
    # 2. Treinar modelo
    model, historico = treinar_modelo(treino_dataset, val_dataset)
    
    # 3. Avaliar modelo
    print("\nAvaliação no Conjunto de Validação:")
    features, labels = avaliar_modelo(model, val_dataset)
    
    # 4. Visualizações
    plotar_curvas(historico)
    visualizar_espaco_latente(features, labels)
    
    return model

# Executar o pipeline
if __name__ == "__main__":
    modelo = pipeline_classificacao()