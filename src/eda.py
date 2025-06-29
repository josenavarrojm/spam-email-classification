# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === Configuración ===
DATA_DIR = Path("data")
PLOT_DIR = Path("plots/eda")
PLOT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "spam.csv"

# === Cargar dataset crudo ===
df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")

# === Preprocesamiento inicial ===
df = df[["v1", "v2"]].dropna()
df = df.rename(columns={"v1": "label", "v2": "text"})
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# === Análisis de balance de clases ===
print("\n📊 Distribución de clases:")
counts = df["label"].value_counts()
print(counts)

print("\n📊 Proporciones:")
print(counts / counts.sum())

# === Visualización ===
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="label", palette="Set2")
plt.xticks([0, 1], ["Ham", "Spam"])
plt.title("Distribution of SMS Classes")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()

# === Guardar gráfico ===
plot_path = PLOT_DIR / "class_distribution.png"
plt.savefig(plot_path)
print(f"📈 Gráfico guardado en: {plot_path}")
