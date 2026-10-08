"""Seasonal-naive forecasts repeat the most recent seven observed days."""
import numpy as np
import pandas as pd


def seasonal_naive(history: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    """Forecast one hospital-medicine series, using historical values only."""
    if horizon_days not in (7, 14):
        raise ValueError("horizon_days must be 7 or 14")
    series = history.sort_values("date")
    if len(series) < 7:
        raise ValueError("Seasonal baseline needs at least seven historical days")
    if len(series[["hospital_id", "medicine_id"]].drop_duplicates()) != 1:
        raise ValueError("Baseline expects exactly one hospital-medicine series")
    dates = pd.date_range(series.date.iloc[-1] + pd.Timedelta(days=1), periods=horizon_days)
    values = np.resize(series.quantity_requested.iloc[-7:].to_numpy(), horizon_days)
    return pd.DataFrame({"date": dates, "predicted_demand": values})
