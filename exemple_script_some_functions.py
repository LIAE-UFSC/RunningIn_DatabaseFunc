from TimeFrequencyRunIn import FFT_ensaio, STFT_ensaio, bandpower
from runInDB_utils import RunIn_File
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy

path = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelA.hdf5"
unit = "A1"
test = "2019_07_01"
N_Samples = 25600
f_s = 25600
index = range(1)



with RunIn_File(path) as file:

    fft = FFT_ensaio(file,unit,test,0) #ta dando problema no último parâmetro quando usa o arquivo
    stft = STFT_ensaio(file,unit,test,index)
    test = file["A1"][0]
    print(bandpower(test,25600,1000,1100))

