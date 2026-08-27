import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(100)

dataset = '../data/classification2.csv'
#dataset = 'diabetes.csv'

data = pd.read_csv(dataset, header=None)

losses = []
weights = []
biases = []
accuracies = []

learning_rate = 0.05
epochs = 10000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def normalize(X):
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

def initialize_weights(input_size, output_size):
    limit = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-limit, limit, (input_size, output_size))

def cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred ))


X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values.reshape(-1, 1) 

X = normalize(X)

train_size = int(0.75* X.shape[0])
indices = np.random.permutation(X.shape[0])

X_train = X[indices[:train_size]]
y_train = y[indices[:train_size]]
X_test = X[indices[train_size:]]
y_test = y[indices[train_size:]]

nn = [X_train.shape[1], 3, 2 ,1]  # arquitetura da rede neural

# for i in range(len(nn) - 1):
#     weights.append(initialize_weights(nn[i], nn[i + 1]))
#     biases.append(np.zeros((1, nn[i + 1])))

# Treinamento da rede neural

# Carregar os pesos treinados
weights = [
    np.load('../models/pytorch_weights_layer0.npy'),  # Camada 1 (2→3)
    np.load('../models/pytorch_weights_layer1.npy'),  # Camada 2 (3→2)
    np.load('../models/pytorch_weights_layer2.npy')   # Camada 3 (2→1)
]

biases = [
    np.load('../models/pytorch_bias_layer0.npy'),  # Bias camada 1
    np.load('../models/pytorch_bias_layer1.npy'),  # Bias camada 2
    np.load('../models/pytorch_bias_layer2.npy')   # Bias camada 3
]

# Sua função predict() original continuará funcionando normalmente
def predict(X):
    activation = X
    for i in range(len(weights)):
        z = np.dot(activation, weights[i]) + biases[i]
        activation = 1 / (1 + np.exp(-z))  # Sigmoid
    return activation

def numerical_gradient(X, y, weights, biases, epsilon=1e-5):
    grad_approx = []
    
    for layer in range(len(weights)):
        weight_grad_approx = np.zeros_like(weights[layer])
        bias_grad_approx = np.zeros_like(biases[layer])
        
        for i in range(weights[layer].shape[0]):
            for j in range(weights[layer].shape[1]):
                weights[layer][i, j] += epsilon
                loss1 = cross_entropy(y, predict(X))
                
                weights[layer][i, j] -= 2 * epsilon
                loss2 = cross_entropy(y, predict(X))
                
                weights[layer][i, j] += epsilon
                
                weight_grad_approx[i, j] = (loss1 - loss2) / (2 * epsilon)
        
        for i in range(biases[layer].shape[1]):
            biases[layer][0, i] += epsilon
            loss1 = cross_entropy(y, predict(X))
            
            biases[layer][0, i] -= 2 * epsilon
            loss2 = cross_entropy(y, predict(X))
            
            biases[layer][0, i] += epsilon
            
            bias_grad_approx[0, i] = (loss1 - loss2) / (2 * epsilon)
        
        grad_approx.append((weight_grad_approx, bias_grad_approx))
    
    return grad_approx

# faz a previsao
def predict(X_new):
    activation = X_new
    for i in range(len(weights)):
        z = np.dot(activation, weights[i]) + biases[i]
        activation = sigmoid(z)
    return activation

#########################################################################################

for epoch in range(epochs):

    activations = [X_train]  

    for i in range(len(weights)):

        z = np.dot(activations[i], weights[i]) + biases[i]
        activation = sigmoid(z)
        activations.append(activation)

    error = y_train - activations[-1]  
    loss = cross_entropy(y_train, activations[-1])
    losses.append(loss)

    # Calcula previsões e armazena a acuráciass
    predictions_train = (activations[-1] > 0.5).astype(int)
    accuracy_train = np.mean(predictions_train == y_train)
    accuracies.append(accuracy_train)

    deltas = []  

    # Delta da camada de saída
    delta_output = error * sigmoid_derivative(activations[-1])  # Cálculo do delta da saída
    deltas.append(delta_output)

    # Cálculo do delta da camada oculta
    for i in reversed(range(len(weights) - 1)):

        delta_proxima = deltas[-1]  
        gradiente = delta_proxima.dot(weights[i + 1].T) 
        delta_oculta = gradiente * sigmoid_derivative(activations[i + 1])   
        deltas.append(delta_oculta)

    deltas.reverse()

    # Atualização dos pesos e bias
    for i in range(len(weights)):
        
        ativacoes_anterior = activations[i]      
        deltas_atual = deltas[i]  
        ativacoes_anterior_T = ativacoes_anterior.T 
        gradiente_pesos = np.dot(ativacoes_anterior_T, deltas_atual)  
        weights[i] += gradiente_pesos * learning_rate
        bias_atualizacao = np.sum(deltas_atual, axis=0, keepdims=True)
        biases[i] += bias_atualizacao * learning_rate

    if epoch % 1000 == 0:
        print(f"Perda na época {epoch}: {loss}")
    
        # Validação do gradiente a cada 100 épocas 
        grad_approx = numerical_gradient(X_train, y_train, weights, biases)
        for layer in range(len(weights)):
            gradiente_pesos_calculado = np.dot(activations[layer].T, deltas[layer])

            # Comparar com o gradiente numérico
            weight_diff = np.linalg.norm(gradiente_pesos_calculado - grad_approx[layer][0])
            bias_diff = np.linalg.norm(bias_atualizacao - grad_approx[layer][1])

            #print(f'Diferença no gradiente dos pesos na camada {layer}: {weight_diff}')
            #print(f'Diferença no gradiente dos biases na camada {layer}: {bias_diff}')


#################################################################################################################################

# Teste com os dados de validação
probabilities = predict(X_test)
predictions = (probabilities > 0.5).astype(int)

# Criando uma lista de tuplas com (probabilidade, previsão)
resultados = list(zip(probabilities.flatten(), predictions.flatten()))

TP = np.sum((predictions.flatten() == 1) & (y_test.flatten() == 1))
TN = np.sum((predictions.flatten() == 0) & (y_test.flatten() == 0))
FP = np.sum((predictions.flatten() == 1) & (y_test.flatten() == 0))
FN = np.sum((predictions.flatten() == 0) & (y_test.flatten() == 1))

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

probabilidades = [f"{prob:.3f}" for prob, _ in resultados]
predicoes = [pred for _, pred in resultados]
valores_verdadeiros = y_test.flatten()

print("Probabilidades (3 casas decimais):")
print(probabilidades)

print("Valores Preditos:")
print(np.array(predicoes))

print("Valores Verdadeiros:")
print(valores_verdadeiros)


def plot_combined_decision_boundary_and_cost(X_train, y_train, losses):
    # Definindo limites do gráfico
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Plotando a fronteira de decisão
    Z = predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).astype(int).reshape(xx.shape)

    plt.figure(figsize=(12, 6))

    # Subplot para a fronteira de decisão
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.flatten(), edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Fronteira de Decisão da Rede Neural')

    # Subplot para o custo por época
    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Custo (Perda)', color='blue')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title('Função de Custo Durante o Treinamento')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_cost(losses):
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label='Custo (Perda)', color='blue')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title('Função de Custo Durante o Treinamento')
    plt.legend()
    plt.show()

def plot_accuracy(accuracies):
    plt.figure(figsize=(8, 6))
    plt.plot(accuracies, label='Acurácia', color='green')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.title('Acurácia Durante o Treinamento')
    plt.legend()
    plt.show()

# Exibindo a matriz de confusão e métricas
print("Matriz de Confusão:")
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")


if dataset == '../data/classification2.csv':
    plot_combined_decision_boundary_and_cost(X_train, y_train, losses)
else:
    plot_cost(losses)

plot_accuracy(accuracies)

