"""Evaluation of classic classifiers (logistic regression, SVM, decision tree) and metrics."""

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
    Evaluate classification models with robust handling of dimensionality.

    Parameters:
    - df_treino: DataFrame with training data (must contain the 'anomaly' column)
    - df_teste: Optional - DataFrame with test data (same columns as training)
    - test_size: Test proportion when df_teste is not provided
    - random_state: Seed for reproducibility

    Returns:
    - Dictionary with metrics for each model
    """

    if 'anomaly' not in df_treino.columns:
        raise ValueError("The dataset must contain the 'anomaly' column")

    feature_cols = [col for col in df_treino.columns if col != 'anomaly']
    if len(feature_cols) == 0:
        raise ValueError("No features found (only the 'anomaly' column)")

    if df_teste is not None:
        if set(df_teste.columns) != set(df_treino.columns):
            raise ValueError("Test dataset columns differ from the training set")
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
        raise ValueError(f"Feature mismatch: train has {X_train.shape[1]}, test has {X_test.shape[1]}")

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
            print(f"Error evaluating {nome}: {str(e)}")
            resultados[nome] = {"erro": str(e)}

    return resultados


def avaliar_modelos_completo(df_treino, df_teste=None, test_size=0.3, random_state=42,
                             retornar_modelos=False):
    """Evaluate the same classifiers with metrics suited to imbalanced classes.

    Same data preparation as ``avaliar_modelos``, but reports, in addition to
    accuracy/precision/recall/F1, the more honest metrics for imbalance:
    balanced accuracy, MCC, ROC-AUC and PR-AUC (average precision). Kept as a
    separate function so as not to change ``avaliar_modelos`` (used by ``gridsearch.py``).

    Parameters and input format are identical to ``avaliar_modelos``, with:
    - retornar_modelos (bool): if True, also returns the trained artifacts
      (classifiers, scaler and feature_cols) for out-of-sample inference,
      e.g. applying ``predict_proba`` on the grey zone. Default False (behavior
      identical to before).

    Focuses on the binary case (not run-in × run-in); AUC is computed only when the
    problem is binary and a probability score is available.

    Returns:
    - If retornar_modelos=False: dictionary {model_name: {metrics}}.
    - If retornar_modelos=True: tuple (resultados, artefatos), where artefatos is
      {"modelos": {name: trained_classifier}, "scaler": StandardScaler,
       "feature_cols": [...]}. The classifiers expose ``predict_proba`` and new data
      must be scaled with ``scaler`` before prediction.
    """
    if 'anomaly' not in df_treino.columns:
        raise ValueError("The dataset must contain the 'anomaly' column")

    feature_cols = [col for col in df_treino.columns if col != 'anomaly']
    if len(feature_cols) == 0:
        raise ValueError("No features found (only the 'anomaly' column)")

    if df_teste is not None:
        if set(df_teste.columns) != set(df_treino.columns):
            raise ValueError("Test dataset columns differ from the training set")
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
        raise ValueError(f"Feature mismatch: train has {X_train.shape[1]}, test has {X_test.shape[1]}")

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
    modelos_treinados = {}

    for nome, modelo in modelos.items():
        try:
            modelo.fit(X_train_scaled, y_train)
            modelos_treinados[nome] = modelo
            y_pred = modelo.predict(X_test_scaled)

            # Probability score for the positive class (needed for AUC)
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
            print(f"Error evaluating {nome}: {str(e)}")
            resultados[nome] = {"erro": str(e)}

    if retornar_modelos:
        artefatos = {
            "modelos": modelos_treinados,
            "scaler": scaler,
            "feature_cols": feature_cols,
        }
        return resultados, artefatos

    return resultados
