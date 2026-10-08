"""Explicitly simulated demand; not evidence of clinical forecasting accuracy."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .data_loader import ROOT, load_data, validate_data

DEFAULT_SYNTHETIC = ROOT / "data/processed/demand_history_synthetic.csv"
SEED = 42


def generate_synthetic(days: int = 180, seed: int = SEED,
                       end_date: str = "2026-10-08",
                       catalog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Simulate levels, weekly seasonality, trends, noise and occasional outbreaks."""
    if days < 1:
        raise ValueError("days must be positive")
    if catalog is None:
        catalog = load_data()
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=end_date, periods=days)
    hospitals = sorted(catalog.hospital_id.unique())
    medicines = sorted(catalog.medicine_id.unique())
    names = catalog.groupby("medicine_id").generic_name.first().to_dict()
    rows = []
    for h, hospital in enumerate(hospitals):
        for m, medicine in enumerate(medicines):
            level = (45 + 12 * h) * (0.7 + 0.25 * m)
            slope = rng.uniform(-0.025, 0.06)
            outbreak_days = np.zeros(days, dtype=int)
            for start in rng.choice(days, size=max(1, days // 60), replace=False):
                outbreak_days[start:min(start + 5, days)] = 1
            for i, date in enumerate(dates):
                weekly = 1 + 0.20 * np.sin(2 * np.pi * date.dayofweek / 7)
                expected = (level + slope * i) * weekly * (1 + 0.4 * outbreak_days[i])
                quantity = max(0, round(expected + rng.normal(0, level * 0.06)))
                rows.append(dict(date=date, hospital_id=hospital, medicine_id=medicine,
                    generic_name=names[medicine], quantity_requested=quantity,
                    quantity_dispensed=np.nan, target_type="requested_demand",
                    source="simulation", source_medicine_id=medicine,
                    outbreak_flag=int(outbreak_days[i]), lead_time_days=5,
                    data_origin="synthetic"))
    return validate_data(pd.DataFrame(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()
    if args.output.resolve() == (ROOT / "data/processed/demand_history_clean.csv").resolve():
        parser.error("Cannot overwrite the observed dataset")
    if args.output.exists():
        parser.error("Output already exists; choose a new --output path to preserve datasets")
    data = generate_synthetic(days=args.days, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Generated {len(data)} explicitly synthetic rows at {args.output}")


if __name__ == "__main__":
    main()
