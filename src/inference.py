"""Reusable inference utilities shared by CLI and Streamlit."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .model import BreastCancerNN, FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "breast_cancer_nn.joblib"


def load_bundle(model_path: Path = MODEL_PATH) -> tuple[dict[str, Any], BreastCancerNN]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}. Run `python scripts/train_model.py` first.")
    bundle = joblib.load(model_path)
    if bundle.get("model_type") != "BreastCancerNN":
        raise ValueError("Unexpected model artifact. Rebuild it with `python scripts/train_model.py`.")
    features = bundle.get("features", FEATURES)
    model = BreastCancerNN(len(features))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return bundle, model


def validate_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    X = df[features].apply(pd.to_numeric, errors="coerce")
    bad = X.columns[X.isna().any()].tolist()
    if bad:
        raise ValueError("Missing or non-numeric values in: " + ", ".join(bad))
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Input contains infinite values in model features.")
    return X


def predict_dataframe(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    bundle, model = load_bundle(model_path)
    features = bundle["features"]
    X = validate_features(df, features)
    X_scaled = bundle["scaler"].transform(X).astype(np.float32)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(torch.from_numpy(X_scaled))).numpy()
    threshold = float(bundle["threshold"])
    result = df.copy()
    result["malignant_probability"] = probabilities
    result["prediction"] = np.where(probabilities >= threshold, "MALIGNANT", "BENIGN")
    return result
