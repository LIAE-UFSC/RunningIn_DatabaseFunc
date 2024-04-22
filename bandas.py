from source.runInDB_utils import RunIn_File
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from TimeFrequencyRunIn import FFT_ensaio, STFT_ensaio, bandpower

path = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelA.hdf5"
unit = "A1"
test = "2019_07_01"
N_Samples = 25600
f_s = 25600
index = range(1)


def dividir_em_bandas(k, n):

    with RunIn_File(path) as file:

        test = file["A1"][0]
        dados = test.getMeasurements(varName=["vibrationRAWLateral"], indexes = [0])[0]["vibrationRAWLateral"]
        tamanho_intervalo = k // n
        extremos = []
        bandpowerlist = []
        for i in range(n):
            inicio = i * tamanho_intervalo
            fim = (i + 1) * tamanho_intervalo if i < n - 1 else k
            power = bandpower(dados,25600,inicio,fim)
            bandpowerlist.append(power)
            extremos.append([inicio, fim])
        return (extremos,bandpowerlist)


k = 25
n = 1

print(dividir_em_bandas(k,n))