# autoencoder_runin — Detecção de amaciamento de compressores

Pipeline para **detecção de amaciamento (run-in)** de compressores herméticos a
partir de séries temporais de vazão mássica (`massFlow`), usando um **autoencoder**
para aprender uma representação latente e **classificadores clássicos** (regressão
logística, SVM-RBF, árvore de decisão) para separar *não amaciado (0)* × *amaciado (1)*.

A avaliação usa **validação cruzada por unidade** (deixa-uma-unidade-de-fora): o modelo
é sempre testado em um compressor que **não** participou do treino, medindo generalização.

## Estrutura

```
.                     # raiz do repositório
  paths.py            # caminhos centrais (datasets, outputs)
  gridsearch.py       # busca exploratória de hiperparâmetros (uso original)
  requirements.txt    # dependências fixadas
  datasets/
    raw/              # séries brutas por unidade/ensaio
    processados/      # séries rotuladas (não amaciado × amaciado)
    gerados/          # intermediários
  src/
    preprocessing/    # rotulagem por tempo, janelamento, undersampling
    models/           # autoencoder
    analysis/         # avaliação de classificadores, UMAP
  experiments/        # estudo de avaliação (ver abaixo)
  outputs/
    plots/, heatmaps/ # ablações e curvas
    experiments/      # tabelas e figuras do estudo
```

## Módulos do estudo (`experiments/`)

| Módulo | Papel |
|---|---|
| `config.py` | unidades (A1–A5), hiperparâmetros, grey zone, seeds, caminhos |
| `data.py` | carrega os dados por unidade (`unit_id`, `source`) |
| `verificar_dados.py` | valida invariantes (toda unidade tem as duas classes) |
| `janelamento.py` | janelamento por ensaio + split por unidade (sem vazamento) |
| `verificar_janelamento.py` | valida ausência de vazamento treino/teste |
| `runner.py` | validação cruzada por unidade sobre o latente do autoencoder |
| `baselines.py` | baselines de comparação: features cruas e PCA |
| `tabela.py` | execução multi-seed, agregação e exportação da tabela de resultados |
| `figuras.py` | figuras UMAP por unidade e da grey zone |
| `greyzone.py` | probabilidade de amaciamento ao longo do tempo (grey zone) |

## Como rodar

Requer as dependências de `requirements.txt` (ambiente com `torch`, `scikit-learn`,
`umap-learn`, `pandas`, `matplotlib`). A partir da raiz do repositório:

```bash
# verificações rápidas
python experiments/verificar_dados.py
python experiments/verificar_janelamento.py

# reproduzir tabelas e figuras do estudo
python experiments/reproduzir.py
```

Os hiperparâmetros ficam em `experiments/config.py` (`HIPERPARAMETROS`).

## Dados

> ⚠️ **Dados proprietários.** O conjunto de amaciamento pertence ao LIAE-UFSC e
> **não é distribuído**. Não pode ser publicado nem incluído em qualquer versão
> pública deste repositório.

Ver [DATA_CARD.md](DATA_CARD.md) para a descrição do conjunto (unidades, contagens
por classe, aquisição e definição dos rótulos).
