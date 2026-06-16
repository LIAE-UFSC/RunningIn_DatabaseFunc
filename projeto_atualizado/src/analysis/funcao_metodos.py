from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef
import pandas as pd
import numpy as np

def avaliar_modelos(df_treino, df_teste=None, test_size=0.3, random_state=42):
    """
    Avalia modelos de classificação com tratamento robusto para dimensionalidade
    
    Parâmetros:
    - df_treino: DataFrame com dados de treino (deve conter coluna 'anomaly')
    - df_teste: Opcional - DataFrame com dados de teste (mesmas colunas do treino)
    - test_size: Proporção para teste caso df_teste não seja fornecido
    - random_state: Seed para reproducibilidade
    
    Retorna:
    - Dicionário com métricas para cada modelo
    """
    
    if 'anomaly' not in df_treino.columns:
        raise ValueError("O dataset deve conter a coluna 'anomaly'")
    
    feature_cols = [col for col in df_treino.columns if col != 'anomaly']
    if len(feature_cols) == 0:
        raise ValueError("Nenhuma feature encontrada (apenas coluna 'anomaly')")
    
    if df_teste is not None:
        if set(df_teste.columns) != set(df_treino.columns):
            raise ValueError("Colunas do dataset de teste diferentes do treino")
    else:
        df_treino, df_teste = train_test_split(
            df_treino,
            test_size=test_size,
            random_state=random_state,
            stratify=df_treino['anomaly']
        )
    
    X_train = df_treino[feature_cols].values
    y_train = df_treino['anomaly'].values
    X_test = df_teste[feature_cols].values
    y_test = df_teste['anomaly'].values
    
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"Mismatch de features: Treino tem {X_train.shape[1]}, Teste tem {X_test.shape[1]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    modelos = {
        "regressao_logistica": LogisticRegression(max_iter=1000),
        "SVM(RBF)": SVC(probability=True),
        "arvore_de_decisao": DecisionTreeClassifier()
    }
    
    resultados = {}
    
    for nome, modelo in modelos.items():
        try:
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred)
            
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


def avaliar_modelos_completo(df_treino, df_teste=None, test_size=0.3, random_state=42):
    """Avalia os mesmos classificadores com métricas adequadas a classes desbalanceadas.

    Igual a ``avaliar_modelos`` no preparo dos dados, mas reporta, além de
    acurácia/precisão/recall/F1, as métricas mais honestas para desbalanceamento:
    balanced accuracy, MCC, ROC-AUC e PR-AUC (average precision). Mantida como
    função separada para não alterar ``avaliar_modelos`` (usada pelo ``gridsearch.py``).

    Parâmetros e formato de entrada são idênticos a ``avaliar_modelos``.
    Foca no caso binário (não amaciado × amaciado); AUC é calculado apenas quando
    o problema é binário e há score de probabilidade disponível.

    Retorna:
    - Dicionário {nome_do_modelo: {métricas}}.
    """
    if 'anomaly' not in df_treino.columns:
        raise ValueError("O dataset deve conter a coluna 'anomaly'")

    feature_cols = [col for col in df_treino.columns if col != 'anomaly']
    if len(feature_cols) == 0:
        raise ValueError("Nenhuma feature encontrada (apenas coluna 'anomaly')")

    if df_teste is not None:
        if set(df_teste.columns) != set(df_treino.columns):
            raise ValueError("Colunas do dataset de teste diferentes do treino")
    else:
        df_treino, df_teste = train_test_split(
            df_treino,
            test_size=test_size,
            random_state=random_state,
            stratify=df_treino['anomaly']
        )

    X_train = df_treino[feature_cols].values
    y_train = df_treino['anomaly'].values
    X_test = df_teste[feature_cols].values
    y_test = df_teste['anomaly'].values

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"Mismatch de features: Treino tem {X_train.shape[1]}, Teste tem {X_test.shape[1]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelos = {
        "regressao_logistica": LogisticRegression(max_iter=1000),
        "SVM(RBF)": SVC(probability=True),
        "arvore_de_decisao": DecisionTreeClassifier()
    }

    classes = np.unique(y_test)
    is_binary = len(classes) == 2

    resultados = {}

    for nome, modelo in modelos.items():
        try:
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)

            # Score de probabilidade para a classe positiva (necessário para AUC)
            y_score = None
            if hasattr(modelo, "predict_proba"):
                y_score = modelo.predict_proba(X_test_scaled)[:, 1]
            elif hasattr(modelo, "decision_function"):
                y_score = modelo.decision_function(X_test_scaled)

            cm = confusion_matrix(y_test, y_pred)

            metricas = {
                "Acuracia": float(accuracy_score(y_test, y_pred)),
                "Balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "MCC": float(matthews_corrcoef(y_test, y_pred)),
            }

            if is_binary:
                tn, fp, fn, tp = cm.ravel()
                metricas.update({
                    "Precisao": float(precision_score(y_test, y_pred, zero_division=0)),
                    "Recall": float(recall_score(y_test, y_pred)),
                    "F1_score": float(f1_score(y_test, y_pred)),
                    "ROC_AUC": float(roc_auc_score(y_test, y_score)) if y_score is not None else None,
                    "PR_AUC": float(average_precision_score(y_test, y_score)) if y_score is not None else None,
                    "matriz_de_confusao": {"VP": int(tp), "VN": int(tn), "FP": int(fp), "FN": int(fn)},
                })
            else:
                metricas.update({
                    "Precisao": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                    "Recall": float(recall_score(y_test, y_pred, average='weighted')),
                    "F1_score": float(f1_score(y_test, y_pred, average='weighted')),
                    "matriz_de_confusao": cm.tolist(),
                    "classes": classes.tolist(),
                })

            resultados[nome] = metricas
        except Exception as e:
            print(f"Erro ao avaliar {nome}: {str(e)}")
            resultados[nome] = {"erro": str(e)}

    return resultados
