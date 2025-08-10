from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix)
import pandas as pd
import numpy as np

def avaliar_modelos(caminho_arquivo_treino, caminho_arquivo_teste=None, test_size=0.3, random_state=42):
    """
    Avalia modelos de classificação com tratamento robusto para dimensionalidade
    
    Parâmetros:
    - caminho_arquivo_treino: Caminho para o arquivo CSV de treino
    - caminho_arquivo_teste: Opcional - Caminho para o arquivo CSV de teste
    - test_size: Usado apenas quando não há dataset de teste separado
    - random_state: Seed para reprodutibilidade
    
    Retorna:
    - Dicionário com métricas para cada modelo
    """
    
    # Carregar dados com verificação de colunas
    df_treino = pd.read_csv(caminho_arquivo_treino)
    
    # Verificar se a coluna 'anomaly' existe
    if 'anomaly' not in df_treino.columns:
        raise ValueError("O dataset deve conter a coluna 'anomaly'")
    
    # Garantir que temos pelo menos uma feature além da coluna 'anomaly'
    feature_cols = [col for col in df_treino.columns if col != 'anomaly']
    if len(feature_cols) == 0:
        raise ValueError("Nenhuma feature encontrada (apenas coluna 'anomaly')")
    
    if caminho_arquivo_teste:
        df_teste = pd.read_csv(caminho_arquivo_teste)
        # Verificar se as colunas são consistentes
        if set(df_teste.columns) != set(df_treino.columns):
            raise ValueError("Colunas do dataset de teste diferentes do treino")
    else:
        df_treino, df_teste = train_test_split(
            df_treino, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df_treino['anomaly']
        )
    
    # Separar features e target
    X_train = df_treino[feature_cols].values
    y_train = df_treino['anomaly'].values
    X_test = df_teste[feature_cols].values
    y_test = df_teste['anomaly'].values
    
    # Verificação de dimensionalidade
    n_features_train = X_train.shape[1]
    n_features_test = X_test.shape[1]
    
    if n_features_train != n_features_test:
        raise ValueError(f"Mismatch de features: Treino tem {n_features_train}, Teste tem {n_features_test}")
    
    # Normalização com verificação
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except ValueError as e:
        raise ValueError(f"Erro na normalização: {str(e)}. Verifique a consistência dos dados.")
    
    # Modelos
    modelos = {
        "regressao_logistica": LogisticRegression(max_iter=1000),
        "SVM(RBF)": SVC(probability=True),
        "arvore_de_decisao": DecisionTreeClassifier()
    }
    
    resultados = {}
    
    for nome, modelo in modelos.items():
        try:
            # Treinamento
            modelo.fit(X_train_scaled, y_train)
            
            # Predição
            y_pred = modelo.predict(X_test_scaled)
            
            # Cálculo de métricas
            cm = confusion_matrix(y_test, y_pred)
            
            # Verificar se é problema binário
            classes = np.unique(y_test)
            is_binary = len(classes) == 2
            
            if is_binary:
                tn, fp, fn, tp = cm.ravel()
                metricas = {
                    "Acuracia": float(accuracy_score(y_test, y_pred)),
                    "Precisao": float(precision_score(y_test, y_pred, zero_division=0)),
                    "Recall": float(recall_score(y_test, y_pred)),
                    "F1_score": float(f1_score(y_test, y_pred)),
                    "matriz_de_confusao": {
                        "VP": int(tp),
                        "VN": int(tn),
                        "FP": int(fp),
                        "FN": int(fn)
                    }
                }
            else:
                metricas = {
                    "Acuracia": float(accuracy_score(y_test, y_pred)),
                    "Precisao": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                    "Recall": float(recall_score(y_test, y_pred, average='weighted')),
                    "F1_score": float(f1_score(y_test, y_pred, average='weighted')),
                    "matriz_de_confusao": cm.tolist(),
                    "classes": classes.tolist()
                }
            
            resultados[nome] = metricas
            
        except Exception as e:
            print(f"Erro ao avaliar {nome}: {str(e)}")
            resultados[nome] = {"erro": str(e)}
    
    return resultados