import matplotlib.pyplot as plt
import numpy as np



#plot comentado
plt.xlabel('Index')
plt.ylabel('Frequência')
plt.title('Potência por Banda')
plt.grid(True)
#plt.pcolormesh(stft_transposta)
#plt.colorbar()  # Adiciona uma barra de cores para referência
#plt.show() 
#####################################################################################
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Dados fornecidos
eixo_y = np.array([[0, 250], [250, 500], [500, 750], [750, 1000]])
valores = np.array([20, 12, 45, 30])

# Convertendo os intervalos para o formato adequado para o heatmap

valoor = []
for intervalo in eixo_y:
    valoor.append('{} - {}'.format(intervalo[0], intervalo[1]))

# Criando o DataFrame do Seaborn com os pontos médios
data = {'Ponto Médio do Intervalo': valoor[::-1], 'Valor': valores[::-1]}
df = pd.DataFrame(data)

# Criando o heatmap usando Seaborn
plt.figure(figsize=(8, 4))
sns.heatmap(df.set_index('Ponto Médio do Intervalo'), annot=True, fmt='d', cmap='hot', cbar=True)
plt.title('Potência por banda')
plt.xlabel('Index')
plt.ylabel('intervalos de frequência')
plt.show()

