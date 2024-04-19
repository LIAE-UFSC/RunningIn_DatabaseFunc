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
    tamanho_intervalo = k // n
    extremos = []
    for i in range(n):
        inicio = i * tamanho_intervalo + 1
        fim = (i + 1) * tamanho_intervalo
        extremos.append((inicio, fim))
    return extremos


#print(dividir_em_bandas(10,2))

def dividir_em_intervalos(k, n):

    with RunIn_File(path) as file:

        test = file["A1"][0]
        dados = test.getMeasurements(varName=["vibrationRAWLateral"], indexes = [0])[0]["vibrationRAWLateral"]
        tamanho_intervalo = k // n
        extremos = [0]  # Adiciona o extremo inicial
        hfl = []
        for i in range(1, n):
            extremos.append(i * tamanho_intervalo)
            print(extremos)
            if len(extremos) == 2: 
                value = bandpower(dados,25600,extremos[0],extremos[1])
                hfl.append(value)
            else:
                value = bandpower(dados,25600,(i*tamanho_intervalo),(i+1)*tamanho_intervalo)
                print("soap")
            extremos.append(value)
            print(extremos)
        extremos.append(k)  # Adiciona o extremo final
        return (extremos,hfl)

# Exemplo de uso
k = 500
n = 3
valores_extremos = dividir_em_intervalos(k, n)
print(valores_extremos)
print("ola")

lista = [1,2,3,4]



