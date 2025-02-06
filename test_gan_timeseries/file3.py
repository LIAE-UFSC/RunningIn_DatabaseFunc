import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

def generate_real_data(n_samples, seq_length):
    time = np.linspace(0, 10, seq_length)
    data = np.sin(time) + 0.1 * np.random.normal(size=seq_length)
    data = np.tile(data, (n_samples, 1))
    return data[:, :, np.newaxis]

latent_dim = 100
seq_length = 50
n_samples = 1000
batch_size = 32
epochs = 100  # Número de épocas

real_data = generate_real_data(n_samples, seq_length)
real_data = torch.FloatTensor(real_data)

dataset = TensorDataset(real_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

generator = Generator(latent_dim, seq_length)
discriminator = Discriminator(seq_length)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

for epoch in range(epochs):
    for i, real_samples in enumerate(dataloader):
        real_samples = real_samples[0]
        optimizer_D.zero_grad()
        real_labels = torch.ones(real_samples.size(0), 1)
        fake_labels = torch.zeros(real_samples.size(0), 1)
        real_output = discriminator(real_samples)
        d_loss_real = criterion(real_output, real_labels)
        z = torch.randn(real_samples.size(0), latent_dim)
        fake_samples = generator(z)
        fake_output = discriminator(fake_samples.detach())
        d_loss_fake = criterion(fake_output, fake_labels) 
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()
        output = discriminator(fake_samples)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        optimizer_G.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

def plot_results(real_data, generator, latent_dim, seq_length):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(10, latent_dim)
        generated_data = generator(z).squeeze().numpy()

    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.plot(real_data[i].squeeze().numpy(), label="Real")
        plt.plot(generated_data[i], label="Synthetic")
        plt.legend()
    plt.show()

plot_results(real_data, generator, latent_dim, seq_length)