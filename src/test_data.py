from pathlib import Path
import csv

DATA_DIR = Path("data/processed")


def read_csv(filename):
    with open(
        DATA_DIR / filename,
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return list(csv.DictReader(file))


def test_demand_file_exists():
    assert (
        DATA_DIR / "demand_history_clean.csv"
    ).exists()


def test_demand_has_required_columns():
    rows = read_csv("demand_history_clean.csv")

    assert len(rows) > 0

    required = {
        "date",
        "hospital_id",
        "medicine_id",
        "quantity_requested",
    }

    assert required.issubset(rows[0].keys())


def test_demand_values_are_valid():
    rows = read_csv("demand_history_clean.csv")

    for row in rows:
        assert row["hospital_id"]
        assert row["medicine_id"]
        assert float(row["quantity_requested"]) >= 0


def test_no_duplicate_demand_records():
    rows = read_csv("demand_history_clean.csv")

    keys = [
        (
            row["date"],
            row["hospital_id"],
            row["medicine_id"],
        )
        for row in rows
    ]

    assert len(keys) == len(set(keys))