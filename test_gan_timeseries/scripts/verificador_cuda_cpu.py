import torch

# Verifica se a GPU está disponível
if torch.cuda.is_available():
    print("GPU está disponível!")
    device = torch.device("cuda")  # Usa a GPU
else:
    print("GPU não está disponível, usando CPU.")
    device = torch.device("cpu")  # Usa a CPU