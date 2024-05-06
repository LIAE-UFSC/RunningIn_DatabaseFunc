from TimeFrequencyRunIn import FFT_ensaio, STFT_ensaio, bandpower, dividir_em_bandas
from source.runInDB_utils import RunIn_File
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy

path = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelA.hdf5"

#transformadas
with RunIn_File(path) as file:

    for unit in file:
        for test in unit:
            # Imprime o nome de todos os testes e unidades do database
            print(f"Unidade: {unit} |Teste: {test}")

    unit = "A1"
    test = "2019_07_01"
    N_Samples = 25600
    f_s = 25600
    index = range(1)

    fft = FFT_ensaio(file,unit,test,0) #ta dando problema no último parâmetro quando usa o arquivo
    stft = STFT_ensaio(file,unit,test,index)
    
    
#Leitura em bandas da potência para um caso específico
with RunIn_File(path) as file: 

    test = file["A1"][0]
    dados = test.getMeasurements(varName=["vibrationRAWLateral"], indexes = [0])[0]["vibrationRAWLateral"]

    print(bandpower(dados,25600,1000,1100))
    print(dividir_em_bandas(1000,4,dados))

# NOVA FUNÇÃO

# Entra:
#   path_in: caminho para o database
#   path_out: caminho para o database
#   n_bandas: número de bandas da análise
#   f_max: frequência máxima da análise
#   janela: número de medições por figura
# Biblioteca sugerida: Pillow (PIL)
#   Tentar gerar figuras grayscale em png.

