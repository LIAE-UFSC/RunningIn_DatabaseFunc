from source.runInDB_utils import RunIn_File
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy


def FFT_ensaio(file:RunIn_File, unidade:str, teste:str, index:int):

    # Retorna FFT da vibração lateral no index desejado

    hm = file[unidade][teste]# path ->  unidade -> ensaio -> index
    dados = hm.getMeasurements(varName=["vibrationRAWLateral"], indexes = [index])[0]["vibrationRAWLateral"]#obtenção dos dados
    N = len(dados); fs = N; T = 1/fs; t = np.arange(0,N/fs,T) #manipulação das entradas
    f = np.fft.fftfreq(N,T) #cria fft
    transf = np.fft.fft(dados) #transformada
    tr = np.abs(transf)
    #plot comentado:
    #plt.xlabel('Index')
    #plt.ylabel('Frequência')
    #plt.title('Espectro de Frequência')
    #plt.grid(True)
    #plt.plot(f,tr)
    #plt.plot(f[f>=0],tr[f>=0])
    #plt.show()
    return (f[f>=0],tr[f>=0])

def STFT_ensaio(file:RunIn_File, unidade:str, teste:str, index:list[int]):

    # Retorna Short-Time fourir Transform da vibração lateral no index desejado
    stft = np.empty((0,12800)) # Inicia uma matriz p/ o resultado da STFT

    for ind in index: #realiza a fft por index e insere no nparray
        (f,fft) = FFT_ensaio(file, unidade, teste, ind)
        stft = np.append(stft,[fft],axis = 0) # Roda a FFT, salva os resultados em uma coluna
    global stft_transposta
    stft_transposta = stft.T # deixa no formato desejado

    #plot comentado
    #plt.xlabel('Index')
    #plt.ylabel('Frequência')
    #plt.title('Espectro de Frequência')
    #plt.grid(True)
    #plt.pcolormesh(stft_transposta)
    #plt.colorbar()  # Adiciona uma barra de cores para referência
    #plt.show()
    return (f,stft,stft_transposta)

def bandpower(data, frequencia_base, freq_min, freq_max): #função que extrai a potência por banda
    f, Pxx = scipy.signal.periodogram(data, frequencia_base)
    ind_min = np.argmax(f > freq_min) - 1
    ind_max = np.argmax(f > freq_max) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

# REVER NOME DE VARIÁVEIS E DOCUMENTAÇÃO
def dividir_em_bandas(freq_max, numero_bandas, data):

        tamanho_intervalo = freq_max // numero_bandas #define o tamanho dos intervalos das bandas
        extremos = [] # lista dos extremos das bandas
        bandpowerlist = [] # lista com os valores de potência por banda
        for i in range(numero_bandas):
            inicio = i * tamanho_intervalo
            fim = (i + 1) * tamanho_intervalo if i < numero_bandas - 1 else freq_max
            power = bandpower(data,25600,inicio,fim) # calcula a bandpower por banda
            bandpowerlist.append(power) # adiciona banda à lista de bandas
            extremos.append([inicio, fim]) # adciona valores extremos por banda
        return (extremos,bandpowerlist) #retorna array com a potência por banda e com os valores extremos



# Entra:
#   path_in: caminho para o database
#   path_out: caminho para o database
#   n_bandas: número de bandas da análise
#   f_max: frequência máxima da análise
#   janela: número de medições por figura
# Biblioteca sugerida: Pillow (PIL)
#   Tentar gerar figuras grayscale em png.

def power_image(freq_max, numero_bandas, data):
    matriz_bandas = []
    for measurement in data:
        dados_array = measurement['vibrationRAWLateral'] # Corrigir depois para pegar a array de uma chave genérica
        matriz_bandas.append(dividir_em_bandas(freq_max,numero_bandas,dados_array))

def gera_dataset(path_in,path_out,n_bandas,f_max,window, var = "vibrationRAWLateral"):
     
    with RunIn_File(path_in) as file:
        for unit in file:
             for test in unit:
                ind_max = max([int(str(val)) for val in test._h5ref.keys() if str(val) != "measurements"])
                i_init = 0
                i_final = window
                while i_final<= ind_max:
                    dados = test.getMeasurements(varName=[var], indexes = range(i_init,i_final))

                    dados_bandas = power_image(freq_max=f_max, numero_bandas=n_bandas, data = dados)
                    dividir_em_bandas(1000,4,dados)

                    # AQUI CRIA E SALVA AS IMAGENS
                    #EIXO X =  TEMPO, TAMANHO JANELA
                    #EIXO Y =  FREQUENCIA, HEAT MAP COM O BANDPOWER (LISTA DE LISTAS)

                    i_init = i_final
                    i_final = i_final+window
    
path = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelA.hdf5"
            

            

power_images(path, "asd", f_max = 1000, n_bandas= 20, window = 10)
