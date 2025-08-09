import pandas as pd
import os
from typing import Union, List, Tuple, Optional

def label_multiple_datasets(
    input_paths: Union[str, List[str]],
    grey_start: float,
    class_1_start: float,
    output_dir: str = "labeled_datasets",
    label_column: str = "label",
    grey_label: str = "grey_zone",
    exclude_grey: bool = True,
    save_greyzone: bool = True
) -> Tuple[List[pd.DataFrame], List[Optional[pd.DataFrame]]]:
    """
    Rotula múltiplos datasets com base em dois pontos de tempo críticos.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(input_paths, str):
        if os.path.isdir(input_paths):
            input_paths = [os.path.join(input_paths, f) for f in os.listdir(input_paths) if f.endswith('.csv')]
        else:
            input_paths = [input_paths]
    
    labeled_datasets = []
    greyzone_datasets = []
    
    for input_path in input_paths:
        try:
            # Carregar o dataset, convertendo vírgulas para pontos na coluna 'time'
            df = pd.read_csv(input_path, decimal=',', thousands='.')
            
            if 'time' not in df.columns:
                print(f"Aviso: Arquivo {input_path} não contém coluna 'time' - pulando")
                continue
            
            # Verificar os dados carregados
            print(f"\nDados carregados de {input_path}:")
            print(df.head())
            
            # Garantir que 'time' é numérico
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            
            # Remover linhas com 'time' inválido (NaN)
            df = df.dropna(subset=['time'])
            
            if df.empty:
                print(f"Aviso: Arquivo {input_path} não tem dados válidos após conversão - pulando")
                continue
            
            # Classificação
            df[label_column] = 0  # Classe 0 por padrão
            df.loc[df['time'] >= grey_start, label_column] = grey_label  # Greyzone
            df.loc[df['time'] >= class_1_start, label_column] = 1  # Classe 1
            
            # Salvar datasets
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            labeled_path = os.path.join(output_dir, f"{base_name}_labeled.csv")
            greyzone_path = os.path.join(output_dir, f"{base_name}_greyzone.csv")
            
            # Separar greyzone (se necessário)
            grey_mask = (df[label_column] == grey_label)
            grey_df = df[grey_mask].copy() if save_greyzone else None
            
            if grey_df is not None and save_greyzone:
                grey_df.to_csv(greyzone_path, index=False)
                greyzone_datasets.append(grey_df)
            else:
                greyzone_datasets.append(None)
            
            # Remover greyzone (se necessário)
            if exclude_grey:
                df = df[~grey_mask]
            
            df.to_csv(labeled_path, index=False)
            labeled_datasets.append(df)
            
            print(f"\nProcessado {input_path}:")
            print(f"  - Classe 0: {(df[label_column] == 0).sum()}")
            print(f"  - Greyzone: {(df[label_column] == grey_label).sum()}")
            print(f"  - Classe 1: {(df[label_column] == 1).sum()}")
            print(f"  - Dataset rotulado salvo em: {labeled_path}")
            if grey_df is not None:
                print(f"  - Greyzone salva em: {greyzone_path}")
            
        except Exception as e:
            print(f"Erro ao processar {input_path}: {str(e)}")
            continue
    
    print(f"\nProcessamento concluído:")
    print(f"- {len(labeled_datasets)} datasets rotulados gerados")
    print(f"- {len([g for g in greyzone_datasets if g is not None])} datasets de greyzone gerados")
    return labeled_datasets, greyzone_datasets


if __name__ == "__main__":
    datasets = ["dados_A1.csv","dados_A2.csv" ]  # Substitua pelo seu arquivo
    grey_start = 18000    # Tempo onde a greyzone começa
    class_1_start = 54000 # Tempo onde a Classe 1 começa
    
    labeled_dfs, greyzone_dfs = label_multiple_datasets(
        input_paths=datasets,
        grey_start=grey_start,
        class_1_start=class_1_start,
        output_dir='.',
        exclude_grey=True,
        save_greyzone=True
    )