import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(100)

def dividir_dados(caminho_arquivo, porcentagem_lote=10):
    """Divide os dados em lotes sequenciais mantendo a estrutura original"""
    caminho_arquivo = Path(caminho_arquivo)
    
    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

    df = pd.read_csv(caminho_arquivo, skiprows=1, header=None)
    tempo = df.iloc[:, 0].values.astype(np.float32)
    dados = df.iloc[:, 1].values.astype(np.float32)
    
    # Dividindo em lotes
    tamanho_lote = int(len(dados) * (porcentagem_lote / 100))
    
    lotes_dados = []
    lotes_tempo = []
    
    for inicio in range(0, len(dados), tamanho_lote):
        fim = inicio + tamanho_lote
        lotes_dados.append(torch.tensor(dados[inicio:fim]).unsqueeze(1))
        lotes_tempo.append(torch.tensor(tempo[inicio:fim]).unsqueeze(1))
    
    # Calculando índices para divisão treino/validação (75%/25%)
    split_idx = int(0.75 * len(lotes_dados))
    
    # Concatenando lotes para treino e validação
    x_train = torch.cat(lotes_dados[:split_idx])
    t_train = torch.cat(lotes_tempo[:split_idx])
    x_val = torch.cat(lotes_dados[split_idx:])
    t_val = torch.cat(lotes_tempo[split_idx:])
    
    return {
        "lotes_dados": lotes_dados[:split_idx],  # Lotes apenas de treino
        "lotes_tempo": lotes_tempo[:split_idx],  # Lotes apenas de treino
        "x_train": x_train,
        "t_train": t_train,
        "x_val": x_val,
        "t_val": t_val
    }

class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.kwargs = kwargs

    def compile_model(self, optimizer_fn=optim.Adam, learning_rate=0.001, criterion_fn=nn.MSELoss):
        """Set up the optimizer and loss function."""
        self.optimizer = optimizer_fn(self.parameters(), lr=learning_rate)
        self.criterion = criterion_fn()

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
        hidden_dim = kwargs.get("hidden_dim", 64)
        activation_fn = kwargs.get("activation_fn", nn.ReLU)
        dropout = kwargs.get("dropout", 0.2)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            activation_fn(),
            nn.Dropout(dropout)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            activation_fn(),
            nn.Dropout(dropout)
        )

        self.latent_output = None
        self.loss_function = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        """Define the forward pass for encoding and decoding."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def compile_autoencoder(self, learning_rate=0.001):
        """Compile with optimizer and loss function specifically for reconstruction."""
        super().compile_model(optimizer_fn=torch.optim.Adam, learning_rate=learning_rate, criterion_fn=nn.MSELoss)
        self.loss_function = nn.MSELoss()
    
    def train_model(self, dataset, epochs=10, batch_size=32, shuffle=True):
        """Train the autoencoder on batches."""
        lotes_dados = dataset["lotes_dados"]
        x_val = dataset.get("x_val", None)

        if self.optimizer is None:
            raise ValueError("Optimizer not set. Please call `compile_model` to set it up.")
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            
            # Process each batch
            for lote in lotes_dados:
                if shuffle:
                    indices = torch.randperm(len(lote))
                    lote = lote[indices]
                
                # Process in mini-batches
                for i in range(0, len(lote), batch_size):
                    x_batch = lote[i:i + batch_size]
                    self.optimizer.zero_grad()
                    _, predictions = self(x_batch)
                    loss = self.loss_function(predictions, x_batch)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item() * len(x_batch)
            
            # Calculate average loss
            avg_train_loss = total_loss / sum(len(lote) for lote in lotes_dados)
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
    
    def evaluate_reconstruction(self, x):
        """Evaluate the reconstruction on a test set."""
        self.eval()
        with torch.no_grad():
            _, decoded = self(x)
            loss = self.loss_function(x, decoded)
        return loss.item()

# Configuração e execução principal
if __name__ == "__main__":

    # Parâmetros do modelo
    params = {
        "input_dim": 1,
        "hidden_dim": 64,
        "activation_fn": nn.ReLU,
        "dropout": 0.2
    }

    # Criar e compilar o modelo
    model = Autoencoder(**params)
    lr = 0.02
    model.compile_autoencoder(learning_rate=lr)

    dataset = dividir_dados("meu_arquivo_massflow_A1_csv.csv", porcentagem_lote=10)
    
    print(f"Dados divididos em {len(dataset['lotes_dados'])} lotes de treino")
    print(f"Tamanho do conjunto de treino: {len(dataset['x_train'])} amostras")
    print(f"Tamanho do conjunto de validação: {len(dataset['x_val'])} amostras")

    # Configurações de treinamento
    epochs = 200
    batch_size = 32

    # Manter cópias dos dados originais
    x_train_original = dataset["x_train"].clone()
    t_train_original = dataset["t_train"].clone()

    #Treinar o modelo
    # print("\nIniciando treinamento...")
    # model.train_model(
    #     dataset=dataset,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     shuffle=False
    # )

    model_path = "autoencoder_weights.pth"
    model.save_model(model_path)
    print(f"Pesos do modelo salvos em {model_path}")

    # Criar novo modelo com mesma arquitetura
    loaded_model = Autoencoder(**params)
    loaded_model.compile_autoencoder(learning_rate=lr)
    
    # Carregar os pesos salvos
    loaded_model.load_model(model_path)
    print("Modelo carregado com sucesso!")
    print(f"Pesos do modelo salvos em {model_path}")

    print("\nPesos salvos no modelo:")
    for name, param in model.named_parameters():
        print(f"\nCamada: {name}")
        print(f"Tamanho: {param.size()}")
        print(f"Valores (primeiros 5): {param.data.flatten()[:5].numpy()}")  # Mostra apenas os primeiros 5 valores

    model.eval()  
    loaded_model.eval()  
    
    with torch.no_grad():

        # Testar com os mesmos dados
        test_data = x_train_original[:5]
        _, original_output = model(test_data)
        _, loaded_output = loaded_model(test_data)
        
        # ve o erro
        difference = torch.mean(torch.abs(original_output - loaded_output)).item()
        print("\nDiferença nas saídas após carregamento:", difference)

    # Avaliar o modelo após o treinamento
    with torch.no_grad():  
        latent_representation = model.encoder(x_train_original)  

    latent_representation = latent_representation.numpy()
    print("\nRepresentação latente obtida:", latent_representation.shape)

    # Visualização com PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_representation)

    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=t_train_original.numpy(), cmap='viridis', alpha=0.6)
    plt.colorbar(label='Tempo')
    plt.title("Representação 2D do Espaço Latente (PCA)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.grid(True)
    plt.show()



    