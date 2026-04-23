from sklearn.decomposition import PCA
import numpy as np

# Criando um conjunto de dados de exemplo (5 amostras, 3 variáveis)
X = np.array([[2.5, 2.4, 3.5],
              [0.5, 0.7, 1.1],
              [2.2, 2.9, 3.1],
              [1.9, 2.2, 2.8],
              [3.1, 3.0, 3.9]])

# Criando um PCA para reduzir para 2 componentes principais
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("Dados transformados:\n", X_reduced)
