from TimeFrequencyRunIn import FFT_ensaio, STFT_ensaio, bandpower, dividir_em_bandas
from plot_bandpower_seaborn import plot_seaborn_bandpower
from source.runInDB_utils import RunIn_File
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy

path = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelA.hdf5"

with RunIn_File(path) as file: 

    test = file["A1"][0]
    bandpowerlista = []

    for k in range(10):
    #for k in range(5):
        dados = test.getMeasurements(varName=["vibrationRAWLateral"], indexes = [k])[0]["vibrationRAWLateral"]
        bandpowerlista.append(dividir_em_bandas(1000,100,dados)[1])
        #bandpowerlista.append(dividir_em_bandas(1000,5,dados)[1])
    
    escala = (dividir_em_bandas(1000,100,dados)[0])
    #escala = (dividir_em_bandas(1000,5,dados)[0])

    #print(bandpowerlista)
    #print(valor)

print(np.min(bandpowerlista))
print(np.max(bandpowerlista))

plot_seaborn_bandpower(escala,bandpowerlista,np.max(bandpowerlista))

#transformadas
with RunIn_File(path) as file:

    #for unit in file:
        #for test in unit:
            # Imprime o nome de todos os testes e unidades do database
            #print(f"Unidade: {unit} |Teste: {test}")

    unit = "A1"
    test = "2019_07_01"
    N_Samples = 25600
    f_s = 25600
    index = range(1)

    fft = FFT_ensaio(file,unit,test,0) #ta dando problema no último parâmetro quando usa o arquivo
    stft = STFT_ensaio(file,unit,test,index)

