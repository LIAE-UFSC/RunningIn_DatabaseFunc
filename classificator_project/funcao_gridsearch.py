import os
import itertools
from autoeoncoderNNpy import *
from funcao_rotulos import label_dataset_by_time
from funcao_janelamento import reorganizar_dataset
from funcao_random_undersampling import balancear_csv_por_undersampling

import os
import itertools

# Cria uma pasta de teste
os.makedirs("pasta_teste", exist_ok=True)

# Gera combinações simples
combos = list(itertools.product([1, 2], ['a', 'b']))
print(combos)