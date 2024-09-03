import cv2
import numpy as np
from source.runInDB_utils import RunIn_File
from TimeFrequencyRunIn import dividir_em_bandas

np.random.seed(10)

#valores = np.random.randint(0, 100, size=(5,5))  

valoresk = np.array([[100, 15, 64, 28, 2], 
                    [93, 29, 8, 73, 0],
                    [40, 36, 16, 11, 54],
                    [88, 62, 33, 72, 78],
                    [49, 51, 54, 77, 69]])

valores = np.array([[1.6246818500095383e-10, 2.0816816308863095e-12, 5.404082588589641e-12, 1.1637742986763643e-12, 7.283843000090124e-13], 
                    [8.078745709762834e-08, 3.955823868143926e-10, 1.5661536903020524e-08, 3.281452846746365e-11, 1.4242067216247294e-10], 
                    [7.941560180929583e-08, 3.1434100435177924e-10, 1.8915640062701137e-08, 3.93355810289904e-11, 4.133915794907403e-10], 
                    [7.143723259442698e-08, 3.1000577103171087e-10, 5.066647980958981e-08, 7.737461238450063e-11, 7.56713307334195e-10], 
                    [7.337697517457091e-08, 3.3615952923621015e-10, 6.723199326051924e-08, 7.122774809243955e-11, 1.0496390103813047e-09]])

matriz_transposta = np.transpose(valores)

valor_minimo = 0
valor_maximo = np.max(valores)

print(valor_maximo)

min_grayscale = 0
max_grayscale = 255

imagem_greyscale = np.interp(valores, (valor_minimo, valor_maximo), (min_grayscale, max_grayscale)).astype(np.uint8) #normal
imagem_greyscale_transposta = np.interp(matriz_transposta, (valor_minimo, valor_maximo), (min_grayscale, max_grayscale)).astype(np.uint8) #transposta

dim = (300, 300)

imagem_greyscale_resized = cv2.resize(imagem_greyscale, dim, interpolation=cv2.INTER_AREA) # maior normal
imagem_greyscale_resized_transposta = cv2.resize(imagem_greyscale_transposta, dim, interpolation=cv2.INTER_AREA) #maior transposta


#cv2.imshow('Imagem Greyscale', imagem_greyscale)
#cv2.imshow('Imagem Greyscale', imagem_greyscale_resized) #maior para ver
#cv2.imshow('Imagem Greyscale', imagem_greyscale_transposta)
cv2.imshow('Imagem Greyscale', imagem_greyscale_resized_transposta) #maior para ver
cv2.waitKey(0)
cv2.destroyAllWindows()


################################## teste com dados reais #########################################

path = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelA.hdf5"

with RunIn_File(path) as file: 

    test = file["A1"][0]
    bandpowerlista = []

    for k in range(5):
        dados = test.getMeasurements(varName=["vibrationRAWLateral"], indexes = [k])[0]["vibrationRAWLateral"]
        bandpowerlista.append(dividir_em_bandas(1000,5,dados)[1])
    
    print(bandpowerlista)

