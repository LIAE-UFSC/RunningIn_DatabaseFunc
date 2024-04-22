from source.runInDB_utils import RunIn_File
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy


def FFT_ensaio(file:RunIn_File, unidade:str, teste:str, index:int):

    # Retorna FFT da vibração lateral no index desejado

    hm = file[unidade][teste]# path ->  unidade -> ensaio -> index

    dados = hm.getMeasurements(varName=["vibrationRAWLateral"], indexes = [index])[0]["vibrationRAWLateral"]
    

    N = len(dados)
    fs = N
    T = 1/fs
    t = np.arange(0,N/fs,T)

    # fft

    f = np.fft.fftfreq(N,T)
    transf = np.fft.fft(dados) #transformada
    tr = np.abs(transf)

    #plot

    plt.xlabel('Index')
    plt.ylabel('Frequência')
    plt.title('Espectro de Frequência')
    plt.grid(True)
    plt.plot(f,tr)
    plt.plot(f[f>=0],tr[f>=0])
    plt.show()
    return (f[f>=0],tr[f>=0])

def STFT_ensaio(file:RunIn_File, unidade:str, teste:str, index:list[int]):

    stft = np.empty((0,12800)) # Inicia uma matriz p/ o resultado da STFT

    for ind in index:
        (f,fft) = FFT_ensaio(file, unidade, teste, ind)
        stft = np.append(stft,[fft],axis = 0) # Roda a FFT, salva os resultados em uma coluna
    global stft_transposta
    stft_transposta = stft.T
    plt.xlabel('Index')
    plt.ylabel('Frequência')
    plt.title('Espectro de Frequência')
    plt.grid(True)
    plt.pcolormesh(stft_transposta)
    plt.colorbar()  # Adiciona uma barra de cores para referência
    plt.show()
    return (f,stft,stft_transposta)

def bandpower(x, fs, fmin, fmax): #essa função estou com dificuldade de execurtar com os dados do dataset, mesmo para apenas um intervalo
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

def dividir_em_bandas(k, n, data):

        tamanho_intervalo = k // n
        extremos = []
        bandpowerlist = []
        for i in range(n):
            inicio = i * tamanho_intervalo
            fim = (i + 1) * tamanho_intervalo if i < n - 1 else k
            power = bandpower(data,25600,inicio,fim)
            bandpowerlist.append(power)
            extremos.append([inicio, fim])
        return (extremos,bandpowerlist)




