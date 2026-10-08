import json
import numpy as np
import pandas as pd
import pytest
from src.ml.baseline import seasonal_naive
from src.ml.evaluate import metrics, evaluate_results
from src.ml.features import FEATURE_NAMES
from src.ml.predict import load_artifact, recursive_forecast
from src.ml.train import train_model


def test_training_artifacts_and_splits(trained):
    artifact, path, directory = trained
    assert path.is_file()
    loaded = load_artifact(path)
    assert loaded["feature_names"] == FEATURE_NAMES
    assert len(loaded["history"]) == 4 * 14
    assert loaded["model"].feature_name_ == FEATURE_NAMES
    report = json.loads((directory / "reports/evaluation.json").read_text())
    assert report["dataset_origin"] == "synthetic"
    assert report["train_date_range"]["end"] < report["validation_date_range"]["start"]
    assert report["validation_date_range"]["end"] < report["test_date_range"]["start"]
    results = pd.read_csv(directory / "reports/prediction_vs_actual.csv")
    assert len(results) == 4 * 14
    assert results.groupby(["hospital_id", "medicine_id"]).size().eq(14).all()
    assert report["lightgbm_metrics"] == metrics(results.actual, results.lightgbm_prediction)
    assert report["baseline_metrics"] == metrics(results.actual, results.baseline_prediction)
    assert set(report["horizons"]) == {"7", "14"}


def test_short_observed_history_rejected(tmp_path):
    from src.ml.data_loader import DEFAULT_DATA
    with pytest.raises(ValueError, match="Insufficient history"):
        train_model(DEFAULT_DATA, tmp_path / "model.joblib", tmp_path / "reports")


def test_zero_wape_and_metrics():
    assert metrics([0, 0], [0, 0])["WAPE"] == 0
    assert metrics([0, 0], [1, 0])["WAPE"] is None
    assert metrics([1, 3], [2, 1]) == {"MAE": 1.5, "WAPE": 0.75, "RMSE": np.sqrt(2.5)}


def test_baseline_repeats_week(synthetic):
    history = synthetic[(synthetic.hospital_id == "HOSP_A") & (synthetic.medicine_id == "PGMED001")]
    result = seasonal_naive(history, 14)
    assert result.predicted_demand.tolist() == history.quantity_requested.iloc[-7:].tolist() * 2
    assert result.date.iloc[0] > history.date.max()


def test_recursive_test_uses_predictions_only(synthetic):
    # A lag-copying model makes leakage visible: every day must reuse the
    # last known demand, even though unseen future actuals are different.
    class LagModel:
        def __init__(self):
            self.seen = []
        def predict(self, features):
            self.seen.append(features.copy())
            return features.lag_1.to_numpy()
    from src.ml.features import category_mappings
    history = synthetic[(synthetic.hospital_id == "HOSP_A") & (synthetic.medicine_id == "PGMED001")].iloc[:40]
    model = LagModel()
    forecast = recursive_forecast(model, history, category_mappings(synthetic), 14)
    assert forecast.predicted_demand.eq(history.quantity_requested.iloc[-1]).all()
    assert len(model.seen) == 14


def test_do_not_claim_improvement():
    result = pd.DataFrame(dict(date=["2026-01-01"], hospital_id=["A"], medicine_id=["M"],
        actual=[10], lightgbm_prediction=[3], baseline_prediction=[10], horizon_day=[1]))
    report = evaluate_results(result, {"dataset_origin": "synthetic"})
    assert report["lightgbm_outperformed_baseline"] is False


def test_holdout_targets_cannot_change_evaluation_forecasts(trained, synthetic, tmp_path):
    changed = synthetic.copy()
    cutoff = changed.date.max() - pd.Timedelta(days=13)
    changed.loc[changed.date >= cutoff, "quantity_requested"] = 10000
    path = tmp_path / "changed.csv"
    changed.to_csv(path, index=False)
    new = train_model(path, tmp_path / "new.joblib", tmp_path / "reports")
    original = trained[0]
    columns = ["date", "hospital_id", "medicine_id", "lightgbm_prediction", "baseline_prediction"]
    pd.testing.assert_frame_equal(original["evaluation_predictions"][columns],
                                  new["evaluation_predictions"][columns])
    assert new["evaluation_metadata"]["best_iteration"] == original["evaluation_metadata"]["best_iteration"]
    assert not new["evaluation_predictions"].actual.equals(original["evaluation_predictions"].actual)
