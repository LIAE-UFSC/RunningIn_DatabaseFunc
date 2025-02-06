import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 1. Preparar os Dados
def generate_real_data(n_samples, seq_length):
    time = np.linspace(0, 10, seq_length)
    data = np.sin(time) + 0.2 * np.random.normal(size=seq_length)  # Série temporal com ruído
    data = np.tile(data, (n_samples, 1))
    return data[:, :, np.newaxis]

# Parâmetros
latent_dim = 100
seq_length = 50
n_samples = 1000
batch_size = 32
epochs = 200  # Número de épocas

# Gerar dados reais
real_data = generate_real_data(n_samples, seq_length)
real_data = torch.FloatTensor(real_data)

# Dataset e DataLoader
dataset = TensorDataset(real_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 2. Criar a GAN
class Generator(nn.Module):
    def __init__(self, latent_dim, seq_length):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, seq_length),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).unsqueeze(-1)

class Discriminator(nn.Module):
    def __init__(self, seq_length):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Instanciar modelos
generator = Generator(latent_dim, seq_length)
discriminator = Discriminator(seq_length)

# Otimizadores
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Função de perda
criterion = nn.BCELoss()

# 3. Treinar a GAN
for epoch in range(epochs):
    for i, real_samples in enumerate(dataloader):
        real_samples = real_samples[0]

        # Treinar o Discriminador
        optimizer_D.zero_grad()
        real_labels = torch.ones(real_samples.size(0), 1)
        fake_labels = torch.zeros(real_samples.size(0), 1)

        # Dados reais
        real_output = discriminator(real_samples)
        d_loss_real = criterion(real_output, real_labels)

        # Dados sintéticos
        z = torch.randn(real_samples.size(0), latent_dim)
        fake_samples = generator(z)
        fake_output = discriminator(fake_samples.detach())
        d_loss_fake = criterion(fake_output, fake_labels)

        # Perda total do Discriminador
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Treinar o Gerador
        optimizer_G.zero_grad()
        output = discriminator(fake_samples)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        optimizer_G.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

# 4. Plotar apenas 2 comparações
def plot_results(real_data, generator, latent_dim, seq_length):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(2, latent_dim)  # Gera 2 vetores de ruído
        generated_data = generator(z).squeeze().numpy()  # Gera 2 séries sintéticas

    plt.figure(figsize=(12, 6))

    # Plot 1: Primeira série real vs. primeira série sintética
    plt.subplot(1, 2, 1)
    plt.plot(real_data[0].squeeze().numpy(), label="Real")
    plt.plot(generated_data[0], label="Synthetic")
    plt.title("Série 1: Real vs. Sintética")
    plt.legend()

    # Plot 2: Segunda série real vs. segunda série sintética
    plt.subplot(1, 2, 2)
    plt.plot(real_data[1].squeeze().numpy(), label="Real")
    plt.plot(generated_data[1], label="Synthetic")
    plt.title("Série 2: Real vs. Sintética")
    plt.legend()

    plt.show()

plot_results(real_data, generator, latent_dim, seq_length)