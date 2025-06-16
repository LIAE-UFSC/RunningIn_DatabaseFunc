import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# carregar o dataset
df = pd.read_csv("dataset_balanceado.csv")

# selecionar automaticamente colunas que começam com 'massFlow'
feature_cols = [col for col in df.columns if col.startswith('massFlow')]
X = df[feature_cols].values
y = df['anomaly'].values

# dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# padronizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# inicializar os modelos
model_logreg = LogisticRegression()
model_svm = SVC(kernel='rbf')
model_tree = DecisionTreeClassifier()

# treinar os modelos
model_logreg.fit(X_train_scaled, y_train)
model_svm.fit(X_train_scaled, y_train)
model_tree.fit(X_train_scaled, y_train)

# fazer previsões
y_pred_logreg = model_logreg.predict(X_test_scaled)
y_pred_svm = model_svm.predict(X_test_scaled)
y_pred_tree = model_tree.predict(X_test_scaled)

# avaliar acurácia
acc_logreg = accuracy_score(y_test, y_pred_logreg)
acc_svm = accuracy_score(y_test, y_pred_svm)
acc_tree = accuracy_score(y_test, y_pred_tree)

# exibir resultados
print(f"Acurácia - Regressão Logística: {acc_logreg:.2f}")
print(f"Acurácia - SVM (RBF): {acc_svm:.2f}")
print(f"Acurácia - Árvore de Decisão: {acc_tree:.2f}")
