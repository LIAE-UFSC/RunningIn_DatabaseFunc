import cv2
import numpy as np

valores = np.random.randint(0, 400, size=(3, 3))  # matriz obtida pelo bandpower

print(valores)

valor_minimo = np.min(valores)
valor_maximo = np.max(valores)

print(valor_maximo)

min_grayscale = 0
max_grayscale = 256

imagem_greyscale = np.interp(valores, (valor_minimo, valor_maximo), (min_grayscale, max_grayscale)).astype(np.uint8)

cv2.imshow('Imagem Greyscale', imagem_greyscale)
cv2.waitKey(0)
cv2.destroyAllWindows()

