import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#função que faz heatmap de bandpower baseado nas listas

def plot_seaborn_bandpower(intervalos,bandpowerlists,max_value):

    intervalos = np.array(intervalos)
    intervalos_str = []

    intervalos_str = ['-'.join(map(str, sublista)) for sublista in intervalos]
    #for intervalo in intervalos:
        #intervalos_str.append('{} - {}'.format(intervalo[0], intervalo[1]))

    data = {'intervalos': intervalos_str[::-1]}
    df = pd.DataFrame(data)

    for k in range ((len(bandpowerlists))):
        valor = bandpowerlists[k]
        df[str(k)] = valor[::-1]

    plt.figure(figsize=(8, 4))
    sns.heatmap(df.set_index('intervalos'),fmt='.30f', cbar=True, vmin= 0, vmax= max_value)
    #annot=True para ver os valores
    plt.title('Potência por banda')
    plt.xlabel('Index')
    plt.ylabel('intervalos de frequência')
    plt.show()


eixo_y = [[0, 250], [250, 500], [500, 750], [750, 1000]]
new_data = [[5,2,3,4],[20, 12, 45, 30],[20, 12, 45, 30],[20, 12, 45, 95]] # Lista de valores bandpower


#plot_seaborn_bandpower(eixo_y,new_data)



