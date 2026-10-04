from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import predict_dataframe  # noqa: E402


def test_sample_prediction_shape_and_labels():
    df = pd.read_csv(ROOT / "data" / "data_breast_cancer.csv").head(5)
    result = predict_dataframe(df)
    assert len(result) == 5
    assert result["malignant_probability"].between(0, 1).all()
    assert set(result["prediction"]).issubset({"MALIGNANT", "BENIGN"})
