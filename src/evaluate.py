import pickle
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve
)

# === Paths ===
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots/evaluation")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# === Load test data ===
with open(DATA_DIR / "X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open(DATA_DIR / "y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

# === Models to evaluate ===
model_files = {
    "LogisticRegression": "LogisticRegression.pkl",
    "MultinomialNB": "MultinomialNB.pkl"
}

# === Store performance results ===
performance = []
roc_data = []

for model_name, filename in model_files.items():
    print(f"🔍 Evaluating {model_name}...")

    # Load model
    with open(MODELS_DIR / filename, "rb") as f:
        model = pickle.load(f)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    performance.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "ROC_AUC": auc
    })

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{model_name}_confusion_matrix.png")
    plt.close()

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data.append((model_name, fpr, tpr, auc))

# === Plot combined ROC curve ===
plt.figure()
for model_name, fpr, tpr, auc in roc_data:
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "roc_comparison.png")
plt.close()

# === Save performance results ===
df_perf = pd.DataFrame(performance)
df_perf.to_csv(MODELS_DIR / "model_performance.csv", index=False)
print("✅ Evaluation completed and results saved.")
