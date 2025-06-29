import pytest
from pathlib import Path

def test_model_files_exist():
    model_dir = Path("models")
    files = ["LogisticRegression.pkl", "MultinomialNB.pkl"]
    for f in files:
        assert (model_dir / f).exists(), f"Missing model file: {f}"
