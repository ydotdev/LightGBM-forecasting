"""Load canonical daily requested demand, rejecting inconsistent records."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "data/processed/demand_history_clean.csv"
KEYS = ["hospital_id", "medicine_id"]
REQUIRED_COLUMNS = {
    "date", "hospital_id", "medicine_id", "generic_name", "quantity_requested",
    "quantity_dispensed", "target_type", "source", "source_medicine_id",
    "outbreak_flag", "lead_time_days",
}


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted copy; never impute, aggregate, or silently repair records."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Demand data is empty")
    data = frame.copy()
    for key in KEYS:
        if data[key].isna().any() or data[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing or empty {key}")
        data[key] = data[key].astype(str)
    try:
        data["date"] = pd.to_datetime(data["date"], format="ISO8601", errors="raise")
        data["quantity_requested"] = pd.to_numeric(data["quantity_requested"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid date or requested demand: {exc}") from exc
    if data.date.isna().any() or data.date.dt.tz is not None:
        raise ValueError("Dates must be nonmissing, timezone-naive daily dates")
    if not data.date.eq(data.date.dt.normalize()).all():
        raise ValueError("Dates must be daily dates without time components")
    demand = data.quantity_requested.to_numpy(dtype=float)
    if not np.isfinite(demand).all() or (demand < 0).any():
        raise ValueError("Requested demand must be finite, nonnegative and nonmissing")
    if data.duplicated(["date", *KEYS]).any():
        raise ValueError("Duplicate (date, hospital_id, medicine_id) records")
    data = data.sort_values(["date", *KEYS]).reset_index(drop=True)
    for ids, series in data.groupby(KEYS, observed=True):
        expected = pd.date_range(series.date.min(), series.date.max(), freq="D")
        gaps = expected.difference(series.date)
        if len(gaps):
            raise ValueError(f"Missing dates for {ids}: {gaps.strftime('%Y-%m-%d').tolist()}")
    return data


def load_data(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    """Read a CSV using string canonical IDs (never source_medicine_id joins)."""
    return validate_data(pd.read_csv(path, dtype={key: str for key in KEYS}))
