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
        dados = test.getMeasurements(varName=["vibrationRAWLateral"], indexes = [k])[0]["vibrationRAWLateral"]

        bandpowerlista.append(dividir_em_bandas(1000,100,dados)[1])
    escala = (dividir_em_bandas(1000,100,dados)[0])
    #print(bandpowerlista)
    #print(valor)

print(np.min(bandpowerlista))
print(np.max(bandpowerlista))

plot_seaborn_bandpower(escala,bandpowerlista,np.max(bandpowerlista))