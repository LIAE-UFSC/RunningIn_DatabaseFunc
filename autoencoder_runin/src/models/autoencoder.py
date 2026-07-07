"""Modelo autoencoder e utilitários de treino, reconstrução e projeção no espaço latente."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
from paths import DATASETS_GER

np.random.seed(42)  

class BaseModel(nn.Module):
    """Classe base para modelos PyTorch, com utilitários de compilação, treino,
    avaliação, predição e persistência (salvar/carregar) compartilhados."""

    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        self.model = None  # Placeholder for model architecture in subclasses
        self.optimizer = None
        self.criterion = None
        self.kwargs = kwargs

    def compile_model(self, optimizer_fn=optim.Adam, learning_rate=0.001, criterion_fn=nn.MSELoss):
        """Set up the optimizer and loss function."""
        self.optimizer = optimizer_fn(self.parameters(), lr=learning_rate)
        self.criterion = criterion_fn()
        # print("Model compiled with custom optimizer and loss function.")

    def train_model(self, x_train, y_train, loss_function, epochs=10, batch_size=32, validation_split=0.2):
        """Train the model on the provided dataset."""
        dataset_size = len(x_train)
        split = int(dataset_size * (1 - validation_split))
        train_data = x_train[:split], y_train[:split]
        val_data = x_train[split:], y_train[split:]

        for epoch in range(epochs):

            self.train()
            total_loss = 0
            for i in range(0, split, batch_size):
                x_batch = train_data[0][i:i + batch_size]
                y_batch = train_data[1][i:i + batch_size]

                self.optimizer.zero_grad()
                predictions = self(x_batch)
                loss = loss_function(predictions, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # Validation
            if validation_split > 0:
                self.eval()
                with torch.no_grad():
                    val_predictions = self(val_data[0])
                    val_loss = loss_function(val_predictions, val_data[1])
                # print(f"Epoch {epoch + 1}, Train Loss: {total_loss:.8f}, Val Loss: {val_loss:.8f}")
            else:
                # print(f"Epoch {epoch + 1}, Loss: {total_loss:.8f}")
                pass

    def evaluate(self, x_test, y_test, loss_function):
        """Evaluate the model on test data."""
        self.eval()
        with torch.no_grad():
            predictions = self(x_test)
            test_loss = loss_function(predictions, y_test)
        print(f"Test loss: {test_loss.item()}")
        return test_loss.item()

    def predict(self, x):
        """Generate predictions on new data."""
        self.eval()
        with torch.no_grad():
            predictions = self(x)
        return predictions

    def save_model(self, file_path):
        """Save the model to a file."""
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """Load the model from a file."""
        self.load_state_dict(torch.load(file_path, weights_only=True))
        print(f"Model loaded from {file_path}")

class Autoencoder(BaseModel):
    """Autoencoder totalmente conectado (encoder/decoder simétricos) com gargalo
    de dimensão ``latent_dim``, treinado por reconstrução (MSE). O encoder produz a
    representação latente usada pelos classificadores."""

    def __init__(self, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        
        input_dim = kwargs.get("input_dim", 1)
        latent_dim = kwargs.get("latent_dim", 32)  # Nova dimensão específica para o bottleneck
        hidden_dim = kwargs.get("hidden_dim", 64)  # Dimensão das camadas intermediárias
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),  # 👈 Camada que define o espaço latente
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.latent_output = None  # Attribute to store the bottleneck layer output
        self.loss_function = nn.MSELoss()  # Set MSE loss as default for Autoencoder

    def forward(self, x):
        """Define the forward pass for encoding and decoding, storing bottleneck output."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def compile_autoencoder(self, learning_rate=0.001):
        """Compile with optimizer and loss function specifically for reconstruction."""
        super().compile_model(optimizer_fn=torch.optim.Adam, learning_rate=learning_rate, criterion_fn=nn.MSELoss)
        self.loss_function = nn.MSELoss()  # Set loss function to MSE for reconstruction
    
    def train_model(self, dataset, epochs=10, batch_size=32, shuffle=True):
        """Train the autoencoder and track loss over epochs."""

        # Extract training data
        x_train = dataset["x_train"]
        # Extract validation data (optional)
        x_val = dataset.get("x_val", None)

        # Initialize lists to store loss values
        self.train_losses = []
        self.val_losses = [] if x_val is not None else None

        # Ensure optimizer is set in compile_model
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Please call `compile_model` to set it up.")
        
        # Shuffle data if requested
        if shuffle:
            indices = torch.randperm(len(x_train))
            x_train = x_train[indices]

        dataset_size = len(x_train)

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for i in range(0, dataset_size, batch_size):
                x_batch = x_train[i:i + batch_size]
                self.optimizer.zero_grad()
                _, predictions = self(x_batch)
                loss = self.loss_function(predictions, x_batch)  # Use the default MSE loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # Store average loss for this epoch
            avg_train_loss = total_loss / dataset_size
            self.train_losses.append(avg_train_loss)

            # Validation
            if x_val is not None:
                self.eval()
                with torch.no_grad():
                    _, val_predictions = self(x_val)
                    val_loss = self.loss_function(val_predictions, x_val).item()
                self.val_losses.append(val_loss)
                print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.8f}, Val Loss: {val_loss:.8f}")
            else:
                print(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.8f}")
                pass
        
    def evaluate(self, x_test, y_test):
        """Evaluate the autoencoder on a test set using reconstruction loss (MSE)."""
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            reconstructed = self(x_test)
            test_loss = self.criterion(reconstructed, y_test)  # MSE loss
        print(f"Test Loss (MSE): {test_loss.item():.8f}")
        return test_loss.item()
    
    def evaluate_reconstruction(self, x):
        """Evaluate the reconstruction on a test set."""
        self.eval()
        with torch.no_grad():
            _, decoded = self(x)
            loss = self.loss_function(x, decoded)
        return loss.item()

    def plot_loss_curve(self, save_path=None):
        """Plot train and validation loss over epochs."""
        if not hasattr(self, 'train_losses') or not self.train_losses:
            raise RuntimeError("No training history found. Train the model first.")

        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(10, 4))
        plt.plot(epochs, self.train_losses, label='Train Loss', color='steelblue')
        if self.val_losses:
            plt.plot(epochs, self.val_losses, label='Val Loss', color='tomato', linestyle='--')
        plt.xlabel('Época')
        plt.ylabel('MSE Loss')
        plt.title('Loss do Autoencoder por Época')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Gráfico salvo em: {save_path}")
        else:
            plt.show()
    

def processar_autoencoder(df_original, params_autoencoder, learning_rate=0.02, epochs=200,
                         batch_size=32, train_size=0.75, df_latente_input=None,
                         return_both_latent=True, return_losses=False):
    """
    Processa dados usando um autoencoder e gera representações latentes
    
    Args:
        df_original (pd.DataFrame): DataFrame com os dados originais para treinamento
        params_autoencoder (dict): Parâmetros para o autoencoder
        learning_rate (float): Taxa de aprendizado
        epochs (int): Número de épocas de treinamento
        batch_size (int): Tamanho do batch
        train_size (float): Proporção dos dados para treinamento
        df_latente_input (pd.DataFrame, optional): DataFrame para gerar o espaço latente
        return_both_latent (bool): Se True, retorna ambos espaços latentes (treino e teste)
        
    Returns:
        tuple: 
            - Se return_both_latent=False: (df_reconstruido, df_latente, dim_latente)
            - Se return_both_latent=True: (df_reconstruido, df_latente_treino, df_latente_teste, dim_latente)
    """
    # --- 1. Divisão dos dados ---
    massflow_cols = [col for col in df_original.columns if col.startswith('massFlow_')]
    if 'anomaly' in massflow_cols:
        massflow_cols.remove('anomaly')
    
    if not massflow_cols:
        raise ValueError("Nenhuma coluna massFlow_* encontrada")
    
    dados = df_original[massflow_cols].values.astype(np.float32)
    tamanho_treino = int(len(dados) * train_size)
    
    dataset = {
        "x_train": torch.tensor(dados[:tamanho_treino]),
        "x_val": torch.tensor(dados[tamanho_treino:])
    }
    
    # --- 2. Configuração do Modelo ---
    model = Autoencoder(**params_autoencoder)
    model.compile_autoencoder(learning_rate=learning_rate)
    
    # --- 3. Treinamento ---
    model.train_model(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False
    )
    
    # --- 4. Reconstrução e Espaço Latente ---
    # Prepara tensores
    dados_tensor = torch.tensor(dados)
    dados_tensor_latente_input = torch.tensor(df_latente_input[massflow_cols].values.astype(np.float32)) if df_latente_input is not None else None
    
    with torch.no_grad():
        # Reconstrução dos dados originais
        _, dados_reconstruidos_tensor = model(dados_tensor)
        
        # Espaço latente do treino
        latent_treino, _ = model(dados_tensor)
        
        # Espaço latente do teste (se existir)
        latent_teste = None
        if dados_tensor_latente_input is not None:
            latent_teste, _ = model(dados_tensor_latente_input)
    
    # --- 5. Preparação dos Resultados ---
    # Reconstrução
    df_reconstruido = df_original.copy()
    for i, col in enumerate(massflow_cols):
        df_reconstruido[col] = dados_reconstruidos_tensor.numpy()[:, i]
    
    # Latente do treino
    df_latent_treino = pd.DataFrame(
        latent_treino.numpy(),
        columns=[f'latent_{i+1}' for i in range(latent_treino.shape[1])]
    )
    if 'anomaly' in df_original.columns:
        df_latent_treino['anomaly'] = df_original['anomaly'].reset_index(drop=True)
    
    # Latente do teste (se existir)
    df_latent_teste = None
    if latent_teste is not None:
        df_latent_teste = pd.DataFrame(
            latent_teste.numpy(),
            columns=[f'latent_{i+1}' for i in range(latent_teste.shape[1])]
        )
        if 'anomaly' in df_latente_input.columns:
            df_latent_teste['anomaly'] = df_latente_input['anomaly'].reset_index(drop=True)
    
    # --- 6. Saída ---
    print(f"Processamento completo! Dimensão latente: {latent_treino.shape[1]}")

    loss_history = {"train": model.train_losses, "val": model.val_losses} if return_losses else None

    if return_both_latent and df_latent_teste is not None:
        base = (df_reconstruido, df_latent_treino, df_latent_teste, latent_treino.shape[1])
    else:
        base = (df_reconstruido, df_latent_treino, latent_treino.shape[1])

    return base + (loss_history,) if return_losses else base

def plot_autoencoder_results(df_original, df_reconstruido, 
                           plot_individual=False, 
                           plot_subplots=False, 
                           plot_agregado=True):
    """
    Gera visualizações diretamente dos DataFrames de saída
    
    Args:
        df_original (pd.DataFrame): DataFrame com dados originais
        df_reconstruido (pd.DataFrame): DataFrame com dados reconstruídos
        plot_individual (bool): Se True, plota cada feature em figuras separadas
        plot_subplots (bool): Se True, plota subplots organizados
        plot_agregado (bool): Se True, plota todas features concatenadas
    """
    # Extrair colunas massFlow
    massflow_cols = [col for col in df_original.columns if col.startswith('massFlow_')]
    
    # Converter para arrays numpy
    original = df_original[massflow_cols].values.astype(np.float32)
    reconstruido = df_reconstruido[massflow_cols].values.astype(np.float32)
    
    num_features = original.shape[1]
    time_steps = np.arange(original.shape[0])
    
    # 1. Plot individual para cada feature
    if plot_individual:
        for i, col in enumerate(massflow_cols):
            plt.figure(figsize=(10, 4))
            plt.plot(original[:, i], label='Original', color='blue', alpha=0.6)
            plt.plot(reconstruido[:, i], label='Reconstruído', color='red', alpha=0.6)
            plt.title(f'Reconstrução da {col}')
            plt.legend()
            plt.show()
    
    # 2. Subplots organizados
    if plot_subplots:
        plt.figure(figsize=(10, 3*num_features))
        for i, col in enumerate(massflow_cols):
            plt.subplot(num_features, 1, i+1)
            plt.plot(time_steps, original[:, i], 'b-', label='Original', alpha=0.7, linewidth=1)
            plt.plot(time_steps, reconstruido[:, i], 'r--', label='Reconstruído', alpha=0.7, linewidth=1)
            plt.title(f'{col} - Original vs Reconstruído')
            plt.ylabel('Valor')
            plt.legend()
            
            if i == num_features-1:
                plt.xlabel('Índice Temporal')
        
        plt.tight_layout()
        plt.show()
    
    # 3. Plot agregado concatenado
    if plot_agregado:
        original_flat = original.flatten()
        reconstruido_flat = reconstruido.flatten()
        time_steps_flat = np.arange(len(original_flat))
        
        plt.figure(figsize=(14, 6))
        plt.plot(time_steps_flat, original_flat, 'b-', 
                label='Original (todas features)', alpha=0.5, linewidth=1)
        plt.plot(time_steps_flat, reconstruido_flat, 'r-', 
                label='Reconstruído (todas features)', alpha=0.5, linewidth=1)
        
        plt.title('Comparação Agregada - Todas Features Concatenadas')
        plt.xlabel('Índice Temporal Contínuo')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    df_original = pd.read_csv(DATASETS_GER / "dataset_balanceado_pronto.csv")
    df_gera_latente = pd.read_csv(DATASETS_GER / "dataset_para_teste_latente.csv")
    
    params_autoencoder = {
        
        "input_dim": 8,
        "hidden_dim": 64,
        "latent_dim": 4,
        "activation_fn": nn.ReLU,
        "dropout": 0.0
    }
    
    # Chamada com retorno dos dois espaços latentes (treino e teste)
    df_reconstruido, df_latente_treino, df_latente_teste, dim_latente = processar_autoencoder(
        df_original=df_original,
        params_autoencoder=params_autoencoder,
        learning_rate=0.02,
        epochs=300,
        batch_size=32,
        train_size=0.75,
        df_latente_input=df_gera_latente,
        return_both_latent=True  # Novo parâmetro para obter ambos
    )
    
    plot_autoencoder_results(
        df_original=df_original,
        df_reconstruido=df_reconstruido,
        plot_individual=False,
        plot_subplots=False,
        plot_agregado=False
    )

    print(df_latente_teste.head())
    print(df_latente_treino.head())
    print(df_reconstruido.head())
    print(df_original.head())