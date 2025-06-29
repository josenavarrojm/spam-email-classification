import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# === Paths ===
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# === Load preprocessed data ===
print("📂 Loading preprocessed data...")
with open(DATA_DIR / "X_train.pkl", "rb") as f:
    X_train = pickle.load(f)
with open(DATA_DIR / "X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open(DATA_DIR / "y_train.pkl", "rb") as f:
    y_train = pickle.load(f)
with open(DATA_DIR / "y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

# === Define models ===
print("⚙️ Initializing models...")
models = {
    "LogisticRegression": LogisticRegression(class_weight="balanced",max_iter=1000),
    "MultinomialNB": MultinomialNB()
}

# === Train, evaluate, and save models ===
results = []

for name, model in models.items():
    print(f"🚀 Training {name}...")
    model.fit(X_train, y_train)

    # Save trained model
    model_path = MODELS_DIR / f"{name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ {name} saved to {model_path}")

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "F1_Score": round(f1, 4)
    })

# === Save performance metrics ===
performance_df = pd.DataFrame(results)
performance_csv = MODELS_DIR / "model_performance.csv"
performance_df.to_csv(performance_csv, index=False)
print(f"📊 Model performance saved to {performance_csv}")
