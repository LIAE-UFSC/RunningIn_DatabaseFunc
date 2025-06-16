import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# carregar o dataset
df = pd.read_csv("AA_espaco_latente.csv")

# selecionar colunas que começam com 'massFlow'
feature_cols = [col for col in df.columns if col.startswith("massFlow")]
X = df[feature_cols].values
y = df['anomaly'].values

# dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# padronizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# modelos
modelos = {
    "Regressão Logística": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(probability=True),
    "Árvore de Decisão": DecisionTreeClassifier()
}

# avaliação
for nome, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    
    print(f"\n=== {nome} ===")
    print(f"Acurácia:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precisão:  {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Revocação: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))


