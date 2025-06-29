import pytest
from src.predict import preprocess_and_predict

def test_prediction_output():
    sample_text = ["Congratulations! You've won a free iPhone. Click now."]
    result = preprocess_and_predict(sample_text)
    assert isinstance(result, list)
    assert all(label in [0, 1] for label in result)
