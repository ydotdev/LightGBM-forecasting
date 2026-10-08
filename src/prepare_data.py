import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

FILES = {
    "ledger": "flexible_hospital_ledger.csv",
    "shortage": "hospital_shortage_forecast_data.csv",
    "shortage_duplicate": "hospital_shortage_forecast_data (1).csv",
    "catalog": "synthetic_medicines.csv",
}


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(input_dir, output_dir):
    raw = {k: read(input_dir / v) for k, v in FILES.items()}
    a, b, c, m = (
        raw[x] for x in ("ledger", "shortage", "shortage_duplicate", "catalog")
    )
    assert b == c, "Shortage copy differs; inspect manually before processing."
    # Canonical ID applies only to the five observed generic names; formulation and strength are UNKNOWN.
    names = sorted({r["generic_name"].strip() for r in a + b})
    canon = {name: f"PGMED{i:03d}" for i, name in enumerate(names, 1)}
    crosswalk = []
    for source, rows in [("ledger", a), ("shortage", b)]:
        for sid, name in sorted(
            {(r["medicine_id"], r["generic_name"].strip()) for r in rows}
        ):
            crosswalk.append(
                {
                    "source": source,
                    "source_medicine_id": sid,
                    "generic_name": name,
                    "canonical_medicine_id": canon[name],
                    "mapping_level": "generic_name_only",
                    "strength": "UNKNOWN",
                    "dosage_form": "UNKNOWN",
                    "review_required": "true",
                }
            )
    write(output_dir / "medicine_id_crosswalk.csv", list(crosswalk[0]), crosswalk)
    # Catalog IDs are from a separate namespace. Do not map by source ID or infer formulation.
    medicines = [
        {
            "medicine_id": canon[n],
            "generic_name": n,
            "strength": "UNKNOWN",
            "dosage_form": "UNKNOWN",
            "unit_of_measure": "unit_unspecified",
            "identity_review_required": "true",
            "source": "hospital_datasets",
        }
        for n in names
    ]
    write(output_dir / "medicines_clean.csv", list(medicines[0]), medicines)
    write(output_dir / "medicine_catalog_unmatched.csv", list(m[0]), m)
    # Distinct observed demand versus dispensing, not mixed into one training target.
    demand = [
        {
            "date": r["date"],
            "hospital_id": r["facility_id"],
            "medicine_id": canon[r["generic_name"].strip()],
            "generic_name": r["generic_name"],
            "quantity_requested": r["daily_demand"],
            "quantity_dispensed": "",
            "target_type": "requested_demand",
            "source": "shortage",
            "source_medicine_id": r["medicine_id"],
            "outbreak_flag": r["is_outbreak_spike"],
            "lead_time_days": r["lead_time_days"],
        }
        for r in b
    ]
    dispensed = [
        {
            "date": r["date"],
            "hospital_id": r["facility_id"],
            "medicine_id": canon[r["generic_name"].strip()],
            "generic_name": r["generic_name"],
            "quantity_requested": "",
            "quantity_dispensed": r["units_dispensed"],
            "target_type": "dispensed_units",
            "source": "ledger",
            "source_medicine_id": r["medicine_id"],
            "outbreak_flag": "",
            "lead_time_days": r["lead_time_days"],
        }
        for r in a
    ]
    write(
        output_dir / "demand_history_clean.csv",
        list(demand[0]),
        sorted(demand, key=lambda x: (x["hospital_id"], x["medicine_id"], x["date"])),
    )
    write(
        output_dir / "dispensing_history_clean.csv",
        list(dispensed[0]),
        sorted(
            dispensed, key=lambda x: (x["hospital_id"], x["medicine_id"], x["date"])
        ),
    )
    metrics_map = {}
    for r in a:
        k = (r["date"], r["facility_id"])
        v = (r["daily_patients_admitted"], r["daily_syndromic_cases"])
        if k in metrics_map and metrics_map[k] != v:
            raise ValueError(f"Conflicting patient metrics: {k}")
        metrics_map[k] = v
    metrics = [
        {
            "date": d,
            "hospital_id": h,
            "daily_patients_admitted": v[0],
            "daily_syndromic_cases": v[1],
            "source": "ledger",
        }
        for (d, h), v in sorted(metrics_map.items())
    ]
    write(output_dir / "hospital_daily_metrics_clean.csv", list(metrics[0]), metrics)
    # Batch IDs repeat across dates in ledger: retain snapshots and do NOT sum them into current inventory.
    inventory = [
        {
            "snapshot_date": r["date"],
            "hospital_id": r["facility_id"],
            "medicine_id": canon[r["generic_name"].strip()],
            "source_medicine_id": r["medicine_id"],
            "batch_id": r["batch_id"],
            "quantity_on_hand": r["stock_on_hand"],
            "expiry_date": r["batch_expiry_date"],
            "source": "ledger",
            "stock_scope": "batch_snapshot",
        }
        for r in a
    ]
    write(output_dir / "inventory_batch_snapshots.csv", list(inventory[0]), inventory)
    stocks = [
        {
            "snapshot_date": r["date"],
            "hospital_id": r["facility_id"],
            "medicine_id": canon[r["generic_name"].strip()],
            "quantity_on_hand": r["total_current_stock"],
            "oldest_batch_expiry_date": r["oldest_batch_expiry_date"],
            "lead_time_days": r["lead_time_days"],
            "source": "shortage",
            "stock_scope": "aggregate_snapshot",
        }
        for r in b
    ]
    write(output_dir / "inventory_aggregate_snapshots.csv", list(stocks[0]), stocks)
    risks = [
        {
            "date": r["date"],
            "hospital_id": r["facility_id"],
            "medicine_id": canon[r["generic_name"].strip()],
            "source_days_to_stockout": r["days_to_stockout"],
            "source_shortage_warning": r["shortage_warning"],
            "source_expiry_risk_status": r["expiry_risk_status"],
            "source": "shortage",
        }
        for r in b
    ]
    write(output_dir / "source_risk_labels_reference_only.csv", list(risks[0]), risks)
    overlap = set((r["date"], r["facility_id"], r["generic_name"]) for r in a) & set(
        (r["date"], r["facility_id"], r["generic_name"]) for r in b
    )
    report = {
        "source_rows": {k: len(v) for k, v in raw.items()},
        "shortage_copy_identical": b == c,
        "canonical_generic_names": canon,
        "canonical_identity_warning": "Mapped by generic name only; strength, formulation and unit are unknown. Not safe for automatic clinical substitution or real transfers.",
        "overlap_hospital_medicine_date_count": len(overlap),
        "training_target_warning": "Requested demand from shortage file and dispensed units from ledger are distinct targets; never coalesce without validated reconciliation.",
        "source_coverage": {},
        "output_rows": {},
        "limitations": [
            "Shortage history is only 15 days; ledger dispensing history is 30 days.",
            "Catalog has a conflicting ID namespace and implausible combinations; retained separately for review.",
            "Batch snapshots are not equivalent to aggregate inventory and cannot be summed across dates.",
            "No validated MOU, transport, hospital reserve, confirmed supplier deliveries, or full inventory transaction history.",
            "Source risk labels are held out of forecasting features to avoid target leakage.",
        ],
    }
    for key, rows in [("shortage", b), ("ledger", a)]:
        report["source_coverage"][key] = {
            "date_min": min(r["date"] for r in rows),
            "date_max": max(r["date"] for r in rows),
            "hospitals": sorted({r["facility_id"] for r in rows}),
            "generic_medicines": sorted({r["generic_name"] for r in rows}),
        }
    report["output_rows"] = {
        "medicines_clean": len(medicines),
        "demand_history_clean": len(demand),
        "dispensing_history_clean": len(dispensed),
        "hospital_daily_metrics_clean": len(metrics),
        "inventory_batch_snapshots": len(inventory),
        "inventory_aggregate_snapshots": len(stocks),
        "source_risk_labels_reference_only": len(risks),
        "medicine_id_crosswalk": len(crosswalk),
        "medicine_catalog_unmatched": len(m),
    }
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_rows": report["output_rows"],
                "overlap_count": len(overlap),
                "duplicate_copy_identical": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = p.parse_args()
    main(args.input_dir, args.output_dir)
