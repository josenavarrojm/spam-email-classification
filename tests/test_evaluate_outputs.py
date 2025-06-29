import pandas as pd
from pathlib import Path

def test_performance_file():
    performance_file = Path("models/model_performance.csv")
    assert performance_file.exists(), "Performance CSV not found"

    df = pd.read_csv(performance_file)
    required_cols = {"Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"}
    assert required_cols.issubset(df.columns), "Missing columns in performance report"
