"""Batch inference CLI.

Examples:
  python scripts/predict.py data/data_breast_cancer.csv
  python scripts/predict.py my_input.csv --output outputs/predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import predict_dataframe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference with the deployed Neural Network.")
    parser.add_argument("csv", help="Input CSV containing the 30 WDBC model features")
    parser.add_argument("--output", default="outputs/predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    input_path = Path(args.csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = predict_dataframe(pd.read_csv(input_path))
    result.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")
    print(result[["malignant_probability", "prediction"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
