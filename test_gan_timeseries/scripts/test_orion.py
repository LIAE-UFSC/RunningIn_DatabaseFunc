from orion.data import load_signal
from orion import Orion

train_data = load_signal('S-1-train')
print("Dados de treino carregados:")
print(train_data.head())


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

print("Iniciando o treinamento...")
orion.fit(train_data)


new_data = load_signal('S-1-new')
print("Novos dados carregados:")
print(new_data.head())

print("Detectando anomalias...")
anomalies = orion.detect(new_data)

print("Anomalias detectadas:")
print(anomalies)
