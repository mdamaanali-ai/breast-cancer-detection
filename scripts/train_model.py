"""Reproducibly train and save the primary Neural Network model."""
from __future__ import annotations

from pathlib import Path
import random
import sys
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import BreastCancerNN, FEATURES, SEED, set_seed  # noqa: E402

DATA_PATH = ROOT / "data" / "data_breast_cancer.csv"
MODEL_PATH = ROOT / "models" / "breast_cancer_nn.joblib"


def metric_dict(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall / Sensitivity": recall_score(y_true, pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probs),
        "PR-AUC": average_precision_score(y_true, probs),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def main() -> None:
    set_seed(SEED)
    random.seed(SEED)
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].copy()
    y = (df["diagnosis"].astype(str).str.upper() == "M").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=SEED
    )

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train).astype(np.float32)
    Xva = scaler.transform(X_val).astype(np.float32)
    Xte = scaler.transform(X_test).astype(np.float32)

    model = BreastCancerNN(len(FEATURES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(Xtr), torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    Xv, yv = torch.from_numpy(Xva), torch.tensor(y_val.to_numpy(), dtype=torch.float32)

    best_state = None
    best_val = float("inf")
    wait = 0
    patience = 25
    epochs = 300
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(Xt), yt)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_loss = criterion(model(Xv), yv).item()
        history.append({"epoch": epoch + 1, "train_loss": float(loss.item()), "val_loss": float(val_loss)})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        val_probs = torch.sigmoid(model(torch.from_numpy(Xva))).numpy()
        test_probs = torch.sigmoid(model(torch.from_numpy(Xte))).numpy()

    # Choose an operating threshold using validation data only.
    candidates = []
    for threshold in np.linspace(0.10, 0.90, 161):
        m = metric_dict(y_val, val_probs, float(threshold))
        if m["Precision"] >= 0.95:
            candidates.append((float(threshold), m))
    threshold = max(candidates, key=lambda x: (x[1]["Recall / Sensitivity"], x[1]["Specificity"]))[0] if candidates else 0.50
    test_metrics = metric_dict(y_test, test_probs, threshold)

    history_path = ROOT / "outputs" / "training_history.csv"
    history_path.parent.mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    metrics_path = ROOT / "outputs" / "test_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2))

    bundle = {
        "model_type": "BreastCancerNN",
        "state_dict": model.state_dict(),
        "scaler": scaler,
        "features": FEATURES,
        "threshold": threshold,
        "seed": SEED,
        "epochs_trained": epoch + 1,
        "metrics": test_metrics,
        "dataset": "Wisconsin Diagnostic Breast Cancer (WDBC)",
        "medical_disclaimer": "Educational demonstration only; not a medical diagnostic system.",
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    # Refresh the key portfolio plots from the exact final model/run.
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    pred = (test_probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4)); ax.imshow(cm); ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_xticks([0,1], ["Benign","Malignant"]); ax.set_yticks([0,1], ["Benign","Malignant"])
    for i in range(2):
        for j in range(2): ax.text(j, i, int(cm[i,j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(assets / "confusion_matrix.png", dpi=180); plt.close(fig)

    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    fig, ax = plt.subplots(figsize=(6, 4)); RocCurveDisplay.from_predictions(y_test, test_probs, ax=ax); ax.set_title("ROC Curve"); fig.tight_layout(); fig.savefig(assets / "roc_curve.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); PrecisionRecallDisplay.from_predictions(y_test, test_probs, ax=ax); ax.set_title("Precision-Recall Curve"); fig.tight_layout(); fig.savefig(assets / "pr_curve.png", dpi=180); plt.close(fig)
    h = pd.DataFrame(history); fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(h["epoch"], h["train_loss"], label="Train loss"); ax.plot(h["epoch"], h["val_loss"], label="Validation loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training Curve"); ax.legend(); fig.tight_layout(); fig.savefig(assets / "training_curve.png", dpi=180); plt.close(fig)

    print(f"Saved: {MODEL_PATH}")
    print(f"Epochs trained: {epoch + 1}")
    print(f"Decision threshold: {threshold:.3f}")
    for key in ["Accuracy", "Precision", "Recall / Sensitivity", "Specificity", "F1", "ROC-AUC", "PR-AUC"]:
        print(f"{key}: {test_metrics[key]:.4f}")


if __name__ == "__main__":
    main()
