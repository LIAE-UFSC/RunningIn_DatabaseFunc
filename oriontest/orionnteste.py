from orion import Orion
import pandas as pd
import torch

if torch.cuda.is_available():
    print(f"Rodando no GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Rodando no CPU")



# Carregar os dados de séries temporais
input_file = "meu_arquivo_massflow_A1_csv.csv"
df = pd.read_csv(input_file)

# Configuração do modelo TadGAN no Orion

hyperparameters = {
    'orion.primitives.aer.AER#1': {
        'epochs': 5,
        'verbose': True
    }
}

orion = Orion(
    pipeline='aer',
    hyperparameters=hyperparameters
)

orion.fit(df)