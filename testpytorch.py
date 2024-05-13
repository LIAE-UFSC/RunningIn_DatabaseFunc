import torch

# Verifica se o PyTorch está instalado corretamente
if torch.cuda.is_available():
    print("Parabéns! PyTorch está instalado e a GPU está disponível!")
else:
    print("Parabéns! PyTorch está instalado, mas a GPU não está disponível.")

##############################################################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# Definindo uma rede neural simples para segmentação

class AnomalyDetector(nn.Module):
    
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Carregando uma imagem em grayscale
image_path = 'path_to_your_image.jpg'
image = plt.imread(image_path)
image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Normalizando a imagem entre 0 e 1
image /= 255.0

# Instanciando o modelo e carregando pesos pré-treinados
model = AnomalyDetector()
model.load_state_dict(torch.load('anomaly_detector_weights.pth'))

# Aplicando o modelo na imagem
with torch.no_grad():
    output = model(image)

# Convertendo a saída para numpy array e removendo a dimensão de lote
output = output.squeeze().numpy()

# Plotando a imagem de saída
plt.imshow(output, cmap='jet')
plt.colorbar()
plt.title('Anomaly Map')
plt.show()
