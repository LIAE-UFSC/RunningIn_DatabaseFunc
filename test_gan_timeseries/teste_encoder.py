import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# def dividir_dados(caminho_arquivo):
#     caminho_arquivo = Path(caminho_arquivo)
    
#     if not caminho_arquivo.exists():
#         raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

#     df = pd.read_csv(caminho_arquivo, skiprows=1, header=None)
#     dados = df.iloc[:, 1].values.astype(np.float32)
#     treino, validacao = train_test_split(dados, test_size=0.25, random_state=42)
    
#     # Convertendo para tensores do PyTorch e ajustando o formato
#     treino = torch.tensor(treino).unsqueeze(1)  # Formato (n, 1)
#     validacao = torch.tensor(validacao).unsqueeze(1)  # Formato (n, 1)
    
#     return {"x_train": treino, "x_val": validacao}

def dividir_dados(caminho_arquivo):
    caminho_arquivo = Path(caminho_arquivo)
    
    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

    df = pd.read_csv(caminho_arquivo, skiprows=1, header=None)
    dados = df.iloc[:, 1].values.astype(np.float32)
    
    # Dividindo os dados de forma sequencial
    tamanho_treino = int(len(dados) * 0.75)
    treino = dados[:tamanho_treino]
    validacao = dados[tamanho_treino:]
    
    # Convertendo para tensores do PyTorch e ajustando o formato
    treino = torch.tensor(treino).unsqueeze(1)  # Formato (n, 1)
    validacao = torch.tensor(validacao).unsqueeze(1)  # Formato (n, 1)
    
    return {"x_train": treino, "x_val": validacao}

class BaseModel(nn.Module):
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
    def __init__(self, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)

        # Extract parameters from kwargs
        input_dim = kwargs.get("input_dim", 1)
        hidden_dim = kwargs.get("hidden_dim", 32)
        activation_fn = kwargs.get("activation_fn", nn.ReLU)
        dropout = kwargs.get("dropout", 0.2)

        # Encoder: Single layer with batch normalization and dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            activation_fn(),
            nn.Dropout(dropout)  # Dropout for regularization
        )

        # Decoder: Single layer with batch normalization and dropout
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            activation_fn(),
            nn.Dropout(dropout)  # Dropout for regularization
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


params = {
    "input_dim": 1,
    "hidden_dim": 4,
    "activation_fn": nn.ReLU,
    "dropout": 0.2
        }

model = Autoencoder(**params)

lr = 0.02
model.compile_autoencoder(learning_rate=lr)
dataset = dividir_dados("meu_arquivo_massflow_A1_csv.csv")
epochs = 200
batch_size = 32
x_train = dataset["x_train"]  

# Mantendo uma cópia dos dados originais
x_train_original = x_train.clone()

# Treinando o modelo
model.train_model(

    dataset=dataset,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=False
)

# Avaliando o modelo após o treinamento
model.eval()  # Coloca o modelo em modo de avaliação
with torch.no_grad():  
    latent_representation = model.encoder(x_train_original)  

latent_representation = latent_representation.numpy()
print("Representação latente:", latent_representation)

pca = PCA(n_components=2)

latent_2d = pca.fit_transform(latent_representation)

plt.figure(figsize=(8, 6))
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6)
plt.title("Representação 2D do Espaço Latente (PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.show()

# Reconstruindo a partir dos dados originais
with torch.no_grad():
    _, reconstruido = model(x_train_original)  # Obter a saída do decodificador

reconstruido = reconstruido.numpy()


# Gráfico de comparação
plt.figure(figsize=(10, 6))
plt.plot(x_train_original.numpy(), label="Original", alpha=0.7)
plt.plot(reconstruido, label="Reconstruído", alpha=0.7)
plt.title("Comparação entre Dados Originais e Reconstruídos")
plt.xlabel("Índice")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)
plt.show()


