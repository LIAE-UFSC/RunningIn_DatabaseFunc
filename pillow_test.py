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
eixo_y = np.array([0, 250, 500, 750])
valores = np.array([20, 12, 45, 30])

# Criando o DataFrame do Seaborn
data = {'Intervalo': eixo_y, 'Valor': valores}
df = pd.DataFrame(data)

# Criando o heatmap usando Seaborn
plt.figure(figsize=(8, 4))
sns.heatmap(df.set_index('Intervalo'), annot=True, fmt='d', cmap='hot', cbar=True)
plt.title('Valores dos Intervalos')
plt.xlabel('Intervalos')
plt.ylabel('Valores')
plt.show()