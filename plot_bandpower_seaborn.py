import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#função que faz heatmap de bandpower baseado nas listas

def plot_seaborn_bandpower(intervalos,bandpowerlists):

    intervalos_str = []
    for intervalo in intervalos:
        intervalos_str.append('{} - {}'.format(intervalo[0], intervalo[1]))

    data = {'intervalos': intervalos_str[::-1]}
    df = pd.DataFrame(data)

    for k in range ((len(bandpowerlists))):
        valor = bandpowerlists[k]
        df[str(k)] = valor[::-1]

    plt.figure(figsize=(8, 4))
    sns.heatmap(df.set_index('intervalos'), annot=True, fmt='d', cbar=True, vmin=0, vmax=100)
    plt.title('Potência por banda')
    plt.xlabel('Index')
    plt.ylabel('intervalos de frequência')
    plt.show()


eixo_y = np.array([[0, 250], [250, 500], [500, 750], [750, 1000]]) #intervalos de frequência
new_data = [[5,2,3,4],[20, 12, 45, 30],[20, 12, 45, 30],[20, 12, 45, 95]] # Lista de valores bandpower


plot_seaborn_bandpower(eixo_y,new_data)



