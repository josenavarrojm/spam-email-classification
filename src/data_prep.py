import pandas as pd
from pathlib import Path
import kagglehub
import pickle
import shutil
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# === Paths ===
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
csv_path = DATA_DIR / "spam.csv"

# === Download the dataset if not already in data/ ===
if not csv_path.exists():
    print("📥 Downloading dataset from Kaggle...")
    downloaded_path = kagglehub.dataset_download(
        'uciml/sms-spam-collection-dataset',
        path='spam.csv'
    )
    
    # Copy from cache to ./data/
    shutil.copy(downloaded_path, csv_path)
    print(f"✅ Dataset copied to {csv_path}")

# === Load the dataset with proper encoding ===
try:
    print("📄 Loading dataset...")
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# === Clean and rename columns ===
print("🧹 Cleaning dataset...")
df = df[["v1", "v2"]].dropna()
df = df.rename(columns={"v1": "label", "v2": "text"})
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# === Train-test split ===
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === TF-IDF Vectorization ===
print("🔠 Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_df=0.95,
    min_df=2,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# === Save preprocessed data ===
print("💾 Saving preprocessed data...")
with open(DATA_DIR / "X_train.pkl", "wb") as f:
    pickle.dump(X_train_tfidf, f)
with open(DATA_DIR / "X_test.pkl", "wb") as f:
    pickle.dump(X_test_tfidf, f)
with open(DATA_DIR / "y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open(DATA_DIR / "y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)
with open(DATA_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Data preparation completed successfully!")
