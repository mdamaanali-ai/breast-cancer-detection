"""Interactive Streamlit portfolio app."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.inference import load_bundle, predict_dataframe  # noqa: E402

st.set_page_config(page_title="Breast Cancer Detection | ML", page_icon="🩺", layout="wide")

MODEL_PATH = ROOT / "models" / "breast_cancer_nn.joblib"
DATA_PATH = ROOT / "data" / "data_breast_cancer.csv"
COMPARISON_PATH = ROOT / "model_comparison.csv"
ASSETS = ROOT / "assets"


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


@st.cache_resource
def get_model():
    return load_bundle(MODEL_PATH)


st.markdown("# 🩺 Breast Cancer Detection")
st.markdown("### Neural Network • Leakage-safe preprocessing • Responsible evaluation")
st.warning(
    "**Educational demonstration only.** This application is not a medical diagnostic system and must not be used to diagnose, treat, or rule out cancer."
)

if not MODEL_PATH.exists():
    st.error("Model artifact is missing. Run `python scripts/train_model.py` first.")
    st.stop()

bundle, _ = get_model()
metrics = bundle.get("metrics", {})
threshold = float(bundle["threshold"])

with st.sidebar:
    st.header("Project")
    st.write("**Primary model:** PyTorch Neural Network")
    st.write(f"**Decision threshold:** `{threshold:.3f}`")
    st.write(f"**Features:** `{len(bundle['features'])}`")
    st.divider()
    st.caption("Wisconsin Diagnostic Breast Cancer (WDBC) dataset")

home, predict, compare, learn = st.tabs(["Overview", "🔎 Predict", "📊 Compare models", "📚 Methodology"])

with home:
    st.subheader("A complete, reproducible ML workflow")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", pct(metrics.get("Accuracy")))
    c2.metric("Sensitivity", pct(metrics.get("Recall / Sensitivity")))
    c3.metric("Specificity", pct(metrics.get("Specificity")))
    c4.metric("ROC-AUC", f"{metrics.get('ROC-AUC', 0):.3f}")

    st.markdown("""
    **Goal:** classify tumors as **Malignant (M)** or **Benign (B)** from 30 numeric measurements.

    The project deliberately evaluates more than accuracy. Because false negatives matter in a cancer-screening context, the operating threshold is selected on a validation split and the held-out test set is used only for final reporting.
    """)

    left, right = st.columns(2)
    with left:
        st.image(str(ASSETS / "confusion_matrix.png"), caption="Held-out test confusion matrix", use_container_width=True)
    with right:
        st.image(str(ASSETS / "roc_curve.png"), caption="ROC curve", use_container_width=True)

with predict:
    st.subheader("Batch prediction")
    st.caption("Upload a CSV containing the 30 model features. Extra columns such as `id` or `diagnosis` are preserved in the downloadable output.")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    use_sample = st.button("Use included sample dataset") if uploaded is None else False

    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif use_sample:
        df = pd.read_csv(DATA_PATH)
    else:
        df = None

    if df is not None:
        try:
            result = predict_dataframe(df, MODEL_PATH)
            malignant = int((result["prediction"] == "MALIGNANT").sum())
            benign = int((result["prediction"] == "BENIGN").sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows analyzed", len(result))
            c2.metric("Predicted malignant", malignant)
            c3.metric("Predicted benign", benign)
            st.dataframe(result, use_container_width=True, height=420)
            st.download_button(
                "⬇️ Download predictions CSV",
                result.to_csv(index=False).encode("utf-8"),
                file_name="breast_cancer_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except ValueError as exc:
            st.error(str(exc))

with compare:
    st.subheader("Model benchmark")
    st.caption("The benchmark is retained from the original project. The Neural Network remains the primary deployed model so the analysis, artifact, CLI and app tell one consistent story.")
    if COMPARISON_PATH.exists():
        comparison = pd.read_csv(COMPARISON_PATH)
        display = comparison.copy()
        for col in ["Accuracy", "Precision", "Recall / Sensitivity", "Specificity", "F1", "ROC-AUC", "PR-AUC"]:
            if col in display:
                display[col] = display[col].map(lambda x: f"{x:.3f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.bar_chart(comparison.set_index("Model")[["Accuracy", "ROC-AUC", "PR-AUC"]])
    else:
        st.info("Model comparison file not found.")

with learn:
    st.subheader("Methodology")
    st.markdown("""
    **1. Data preparation**  
    The supplied CSV is cleaned by excluding the identifier and empty column. `M` is encoded as the positive class.

    **2. Leakage-safe split**  
    A stratified train/test split is followed by a validation split from the training partition. The `StandardScaler` is fitted only on training data.

    **3. Primary model**  
    A compact PyTorch network uses BatchNorm, ReLU, dropout and AdamW. Early stopping selects the best validation-loss state.

    **4. Threshold selection**  
    The classification threshold is selected on validation data, subject to a precision constraint, then frozen for the held-out test evaluation.

    **5. Evaluation**  
    Accuracy, precision, sensitivity/recall, specificity, F1, ROC-AUC and PR-AUC are reported. Confusion matrix, ROC/PR, calibration and feature-importance visuals are included in the repository.
    """)
    st.image(str(ASSETS / "training_curve.png"), caption="Training and validation behavior", use_container_width=True)
    st.image(str(ASSETS / "pr_curve.png"), caption="Precision-recall curve", use_container_width=True)

st.divider()
st.caption("Portfolio project by Mohammed Amaan Ali • Educational use only • Not clinically validated")
