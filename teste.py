import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt

# 🔹 1. Carregar o dataset (já pré-processado)
df = pd.read_csv("dataset_modelagem_completo.csv")

# 🔹 2. Corrigir coluna 'gender' para valores numéricos
df["gender"] = df["gender"].map({"F": 0, "M": 1})

# 🔹 3. Separar features e target
X = df.drop(columns=["subject_id", "stay_id", "target"])
y = df["target"]

# 🔹 4. Dividir entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 🔹 5. Treinar modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 6. Previsões
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 🔹 7. Avaliação do modelo
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print(f"\n🎯 Accuracy: {acc:.4f}")
print(f"🧠 AUC ROC: {auc:.4f}")
print("\n📊 Classification Report:")
print(report)
print("📉 Matriz de Confusão:")
print(matrix)

# 🔹 8. Importância das variáveis
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
importances.plot(kind="barh")
plt.title("Top 15 Variáveis mais Importantes (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()