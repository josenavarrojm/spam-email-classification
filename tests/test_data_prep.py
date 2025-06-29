import pytest
import pickle
from pathlib import Path

def test_preprocessed_data_exists():
    data_dir = Path("data")
    files = ["X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl", "vectorizer.pkl"]
    for f in files:
        assert (data_dir / f).exists(), f"Missing preprocessed file: {f}"
