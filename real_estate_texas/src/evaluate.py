import argparse
import json
from pathlib import Path

import pandas as pd


def load_metrics(metrics_path: str = "models/metrics.json") -> dict:
    root = Path(__file__).resolve().parents[1]
    path = root / metrics_path
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_metrics(metrics: dict) -> pd.DataFrame:
    rows = []
    for model_name, vals in metrics.items():
        if model_name == "best_model":
            continue
        rows.append(
            {
                "model": model_name,
                "r2": vals.get("r2"),
                "rmse": vals.get("rmse"),
                "mae": vals.get("mae"),
            }
        )
    return pd.DataFrame(rows).sort_values("r2", ascending=False)


def main(metrics_path: str = "models/metrics.json") -> None:
    metrics = load_metrics(metrics_path)
    summary = summarize_metrics(metrics)
    print(summary.to_string(index=False))
    best = metrics.get("best_model", {})
    print(f"\nBest model: {best.get('name')} | R2: {best.get('r2')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model comparison metrics.")
    parser.add_argument("--metrics_path", type=str, default="models/metrics.json")
    args = parser.parse_args()
    main(metrics_path=args.metrics_path)
