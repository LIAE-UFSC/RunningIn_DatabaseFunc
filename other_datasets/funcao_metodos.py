import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix)


def avaliar_modelos(caminho_arquivo, test_size=0.3, random_state=42):
    """
    Avalia modelos de classificação e retorna métricas no formato JSON serializável
    """
    # Carregar e preparar dados
    df = pd.read_csv(caminho_arquivo)
    feature_cols = [col for col in df.columns if col != "anomaly"]
    X = df[feature_cols].values
    y = df['anomaly'].values
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos a serem avaliados
    
    modelos = {
        "regressao_logistica": LogisticRegression(max_iter=1000),
        "SVM(RBF)": SVC(probability=True),
        "arvore_de_decisao": DecisionTreeClassifier()
    }
    
    resultados = {}
    
    for nome, modelo in modelos.items():
        # Treinar e prever
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
        
        # Calcular métricas (convertendo para tipos nativos Python)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        resultados[nome] = {
            "Acuracia": float(accuracy_score(y_test, y_pred)),  # Converter para float
            "Precisao": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred)),
            "F1_score": float(f1_score(y_test, y_pred)),
            "matriz_de_confusao": [
                [int(tn), int(fp)],  # Converter para int
                [int(fn), int(tp)]
            ],
            "Detalhes_Matriz": {
                "verdadeiros_ositivos (VP)": int(tp),
                "verdadeiros_negativos (VN)": int(tn),
                "falsos_positivos (FP)": int(fp),
                "Falsos_negativos (FN)": int(fn)
            }
        }
    
    return resultados

if __name__ == "__main__":
    # resultados = avaliar_modelos("dataset_espaco_latente.csv")
    resultados = avaliar_modelos('dataset_balanceado_pronto.csv')
    
    for modelo, metricas in resultados.items():
        print(f"\n=== {modelo} ===")
        for nome_metrica, valor in metricas.items():
            if nome_metrica != "Matriz de Confusão":
                # Verifica se o valor é numérico antes de formatar
                if isinstance(valor, (int, float)):
                    print(f"{nome_metrica}: {valor:.4f}")
                else:
                    print(f"{nome_metrica}: {valor}")
            else:
                print("\nMatriz de Confusão:")
                print(valor)
    
    # Para retornar os resultados (se necessário para outras operações)

