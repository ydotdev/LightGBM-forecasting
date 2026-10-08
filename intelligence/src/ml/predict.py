"""Load persisted artifacts and predict recursively, without retraining."""
from pathlib import Path
from typing import cast
import joblib
import numpy as np
import pandas as pd
from .data_loader import ROOT, KEYS
from .features import FEATURE_NAMES, build_features

DEFAULT_MODEL = ROOT / "intelligence/models/lightgbm_v1.joblib"


def load_artifact(model_path: str | Path = DEFAULT_MODEL) -> dict:
    """Load a trusted local joblib artifact. Never load untrusted pickle files."""
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model not trained: {model_path}. Run src.ml.train first.")
    artifact = joblib.load(model_path)
    if artifact["feature_names"] != FEATURE_NAMES:
        raise ValueError("Saved model feature definitions do not match this module")
    return artifact


def recursive_forecast(model, history: pd.DataFrame, categories: dict,
                       horizon_days: int) -> pd.DataFrame:
    """Append only model predictions; never read future actual demand."""
    if horizon_days not in (7, 14):
        raise ValueError("horizon_days must be 7 or 14")
    if len(history) < 14 or len(history[KEYS].drop_duplicates()) != 1:
        raise ValueError("Recursive prediction needs one series with at least 14 days")
    cols = ["date", *KEYS, "quantity_requested"]
    working = cast(pd.DataFrame, history[cols]).sort_values("date").tail(14).copy()
    if not working.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Recursive prediction requires consecutive daily history")
    rows = []
    for _ in range(horizon_days):
        last = working.iloc[-1]
        date = last.date + pd.Timedelta(days=1)
        candidate = pd.DataFrame([dict(date=date, hospital_id=str(last.hospital_id),
            medicine_id=str(last.medicine_id), quantity_requested=np.nan)])
        expanded = pd.concat([working, candidate], ignore_index=True)
        features = build_features(expanded, categories).iloc[[-1]][FEATURE_NAMES]
        value = float(model.predict(features)[0])
        if not np.isfinite(value):
            raise ValueError("Model produced nonfinite demand")
        value = max(0.0, value)
        candidate["quantity_requested"] = value
        working = pd.concat([working, candidate], ignore_index=True).tail(14)
        rows.append({"date": date, "predicted_demand": value})
    return pd.DataFrame(rows)


def predict_demand(hospital_id: str, medicine_id: str, horizon_days: int = 7,
                   *, model_path: str | Path = DEFAULT_MODEL,
                   artifact: dict | None = None) -> dict:
    """Return exactly 7 or 14 daily forecasts after this pair's latest history."""
    if horizon_days not in (7, 14):
        raise ValueError("horizon_days must be 7 or 14")
    saved = load_artifact(model_path) if artifact is None else artifact
    history = saved["history"]
    history = history[(history.hospital_id == hospital_id) & (history.medicine_id == medicine_id)]
    if history.empty:
        raise ValueError(f"Unknown hospital-medicine combination: {hospital_id}/{medicine_id}")
    forecast = recursive_forecast(saved["model"], history, saved["categories"], horizon_days)
    return dict(hospital_id=hospital_id, medicine_id=medicine_id,
        model_version=saved["model_version"], horizon_days=horizon_days,
        predictions=[dict(date=row.date.strftime("%Y-%m-%d"),
                          predicted_demand=float(row.predicted_demand))
                     for row in forecast.itertuples()])
