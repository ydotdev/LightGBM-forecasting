"""Compute honest holdout metrics and export reproducible reports."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data_loader import ROOT
from .predict import DEFAULT_MODEL, load_artifact

DEFAULT_REPORTS = ROOT / "intelligence/reports"


def metrics(actual, predicted) -> dict:
    """WAPE is a ratio; undefined for zero total actual demand and nonzero error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Metrics require nonempty, equally shaped observations")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite observations")
    error = np.abs(actual - predicted)
    total = float(np.abs(actual).sum())
    wape = float(error.sum() / total) if total else (0.0 if error.sum() == 0 else None)
    return {"MAE": float(error.mean()), "WAPE": wape,
            "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2)))}


def evaluate_results(results: pd.DataFrame, metadata: dict) -> dict:
    """Evaluate both models on identical pair/date rows; no historical refitting."""
    if results.empty or results.duplicated(["date", "hospital_id", "medicine_id"]).any():
        raise ValueError("Evaluation needs nonempty unique pair/date observations")
    report = dict(metadata)
    report["evaluation_observations"] = len(results)
    report["lightgbm_metrics"] = metrics(results.actual, results.lightgbm_prediction)
    report["baseline_metrics"] = metrics(results.actual, results.baseline_prediction)
    report["lightgbm_outperformed_baseline"] = (
        report["lightgbm_metrics"]["MAE"] < report["baseline_metrics"]["MAE"]
        and all(report["lightgbm_metrics"][key] is not None
                and report["baseline_metrics"][key] is not None
                and report["lightgbm_metrics"][key] <= report["baseline_metrics"][key]
                for key in ("WAPE", "RMSE")))
    report["comparison_rule"] = "Lower MAE and no worse WAPE/RMSE on the same rows"
    report["horizons"] = {}
    for horizon in (7, 14):
        if results.horizon_day.max() >= horizon:
            subset = results[results.horizon_day <= horizon]
            report["horizons"][str(horizon)] = {
                "observations": len(subset),
                "lightgbm": metrics(subset.actual, subset.lightgbm_prediction),
                "baseline": metrics(subset.actual, subset.baseline_prediction)}
    report["limitations"] = (
        "Synthetic metrics measure simulated behavior only; no real-world validation. "
        "One forecast origin, no uncertainty intervals. Outbreak flags are excluded."
        if "synthetic" in report["dataset_origin"] else
        "Single forecast origin; limited evidence, no uncertainty intervals. Outbreak flags excluded.")
    return report


def save_evaluation(results: pd.DataFrame, metadata: dict,
                    report_dir: str | Path = DEFAULT_REPORTS) -> dict:
    """Save actual-vs-prediction rows and JSON with no NaN/Infinity values."""
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = evaluate_results(results, metadata)
    results.to_csv(directory / "prediction_vs_actual.csv", index=False)
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()
    saved = load_artifact(args.model)
    report = save_evaluation(saved["evaluation_predictions"], saved["evaluation_metadata"], args.reports)
    print(json.dumps({key: report[key] for key in (
        "dataset_origin", "lightgbm_metrics", "baseline_metrics", "lightgbm_outperformed_baseline")}, indent=2))


if __name__ == "__main__":
    main()
