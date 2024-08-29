from orion import Orion
import pandas as pd
import torch

if torch.cuda.is_available():
    print(f"Rodando no GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Rodando no CPU")

input_file = "oriontest.csv"
df = pd.read_csv(input_file)

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

