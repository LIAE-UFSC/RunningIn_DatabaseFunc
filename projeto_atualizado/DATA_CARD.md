# Data Card — projeto_atualizado

Descrição do conjunto de dados usado no estudo de detecção de amaciamento.

## Origem

Ensaios de bancada de compressores herméticos (LIAE-UFSC). Cada ensaio registra a
**vazão mássica** (`massFlow`) ao longo do tempo. Unidades identificadas como A1–A5
(modelo/unidade de compressor); cada unidade tem um ou mais ensaios datados.

- Sinal: `massFlow` (vazão mássica).
- Amostragem: ~1 amostra a cada **60 s** (`time` em segundos: 60, 120, 180, …).
- Variável de interesse: estado de amaciamento do compressor.

## Rotulagem

Rótulo `anomaly`: **0 = não amaciado**, **1 = amaciado** (estado físico do compressor).

- Ensaios de compressor **novo** (arquivos `_NA`) amaciam **durante** o ensaio e são
  rotulados por tempo:
  - `t ∈ [0, 18000]` → 0 (não amaciado)
  - `t ∈ [54000, ∞)` → 1 (amaciado)
- Ensaios de compressor **já amaciado** → rótulo 1 no ensaio inteiro.
- **Grey zone** `t ∈ (18000, 54000)`: região de transição, **excluída** do
  treino/avaliação (rótulo ambíguo). Usada só na análise qualitativa da transição.

Parâmetros correspondentes em `experiments/config.py`:
`TIME_RANGES = [(0, 18000, 0), (54000, 500000, 1)]`, `GREY_ZONE = (18000, 54000)`.

## Contagens (datasets processados, grey zone excluída)

| Unidade | Ensaios | Amostras | Classe 0 (não amaciado) | Classe 1 (amaciado) |
|---|---|---|---|---|
| A1 | 1 | 995 | 299 | 696 |
| A2 | 6 | 8 290 | 299 | 7 991 |
| A3 | 3 | 4 131 | 299 | 3 832 |
| A4 | 4 | 10 494 | 299 | 10 195 |
| A5 | 3 | 6 654 | 299 | 6 355 |
| **Total** | **17** | **30 564** | **1 495** | **29 069** |

Observações:
- Todas as unidades têm as **duas classes** (cada uma tem ≥1 ensaio `_NA`), o que
  viabiliza a validação cruzada por unidade nas 5 dobras.
- Forte **desbalanceamento** (classe 0 escassa): tratado por undersampling no treino;
  a avaliação usa métricas robustas a desbalanceamento (balanced accuracy, MCC, AUC).

## Reprodutibilidade

- Seeds fixadas (`experiments/config.py` → `SEEDS`).
- Contagens verificáveis por `python experiments/verificar_dados.py`.
