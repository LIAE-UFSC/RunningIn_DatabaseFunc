import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix)


def avaliar_modelos(caminho_treino, caminho_teste, random_state=42):
    """
    Avalia modelos de classificação usando datasets separados para treino e teste.
    Retorna métricas no formato JSON serializável.
    
    Args:
        caminho_treino: Caminho para o arquivo CSV de treino
        caminho_teste: Caminho para o arquivo CSV de teste
        random_state: Seed para reprodutibilidade
    """
    # Carregar dados de treino e teste
    df_treino = pd.read_csv(caminho_treino)
    df_teste = pd.read_csv(caminho_teste)
    
    # Verificar se as colunas são iguais em ambos datasets
    if not set(df_treino.columns) == set(df_teste.columns):
        raise ValueError("Os datasets de treino e teste têm colunas diferentes")
    
    # Preparar features e target
    feature_cols = [col for col in df_treino.columns if col != "anomaly"]
    X_train = df_treino[feature_cols].values
    y_train = df_treino['anomaly'].values
    X_test = df_teste[feature_cols].values
    y_test = df_teste['anomaly'].values
    
    # Normalizar dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos a serem avaliados
    modelos = {
        "regressao_logistica": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVM(RBF)": SVC(probability=True, random_state=random_state),
        "arvore_de_decisao": DecisionTreeClassifier(random_state=random_state)
    }
    
    resultados = {}
    
    for nome, modelo in modelos.items():
        # Treinar e prever
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
        
        # Calcular métricas (convertendo para tipos nativos Python)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        resultados[nome] = {
            "Acuracia": float(accuracy_score(y_test, y_pred)),
            "Precisao": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred)),
            "F1_score": float(f1_score(y_test, y_pred)),
            "matriz_de_confusao": [
                [int(tn), int(fp)],
                [int(fn), int(tp)]
            ],
            "Detalhes_Matriz": {
                "verdadeiros_positivos (VP)": int(tp),
                "verdadeiros_negativos (VN)": int(tn),
                "falsos_positivos (FP)": int(fp),
                "falsos_negativos (FN)": int(fn)
            }
        }
    
    return resultados


if __name__ == "__main__":
    
    # Exemplo de uso
    resultados = avaliar_modelos(
        caminho_treino='dataset_balanceado_pronto.csv',
        caminho_teste='dataset_para_teste.csv'
    )
    
    # Exibir resultados
    for modelo, metricas in resultados.items():
        print(f"\n=== {modelo} ===")
        for nome_metrica, valor in metricas.items():
            if nome_metrica != "matriz_de_confusao":
                if isinstance(valor, (int, float)):
                    print(f"{nome_metrica}: {valor:.4f}")
                else:
                    print(f"{nome_metrica}: {valor}")
            else:
                print("\nMatriz de Confusão:")
                print(f"[[TN: {valor[0][0]}, FP: {valor[0][1]}]")
                print(f" [FN: {valor[1][0]}, TP: {valor[1][1]}]]")