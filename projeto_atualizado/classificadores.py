import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def avaliar_classificadores(df_treino, df_teste, target_col='anomaly'):
    """
    Avalia classificadores usando datasets separados para treino e teste
    
    Args:
        df_treino (pd.DataFrame): DataFrame com dados de treino
        df_teste (pd.DataFrame): DataFrame com dados de teste
        target_col (str): Nome da coluna alvo (default: 'anomaly')
    """
    # Separa features e target
    feature_cols = [col for col in df_treino.columns if col != target_col]
    
    X_train = df_treino[feature_cols].values
    y_train = df_treino[target_col].values
    
    X_test = df_teste[feature_cols].values
    y_test = df_teste[target_col].values
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos
    modelos = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "SVM (RBF)": SVC(probability=True),
        "Árvore de Decisão": DecisionTreeClassifier()
    }
    
    resultados = {}
    
    for nome, modelo in modelos.items():
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
        
        # Calcula métricas
        metricas = {
            'Acurácia': accuracy_score(y_test, y_pred),
            'Precisão': precision_score(y_test, y_pred, zero_division=0),
            'Revocação': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred),
            'Matriz de Confusão': confusion_matrix(y_test, y_pred)
        }
        
        resultados[nome] = metricas
        
        # Exibe resultados (igual à versão original)
        print(f"\n=== {nome} ===")
        print(f"Acurácia:  {metricas['Acurácia']:.4f}")
        print(f"Precisão:  {metricas['Precisão']:.4f}")
        print(f"Revocação: {metricas['Revocação']:.4f}")
        print(f"F1-score:  {metricas['F1-score']:.4f}")
        print("Matriz de Confusão:")
        print(metricas['Matriz de Confusão'])
    
    return resultados

if __name__ == "__main__":

    # Carrega datasets separados
    df_treino = pd.read_csv("dataset_treino.csv")
    df_teste = pd.read_csv("dataset_teste.csv")

    # Avalia classificadores
    resultados = avaliar_classificadores(df_treino, df_teste)