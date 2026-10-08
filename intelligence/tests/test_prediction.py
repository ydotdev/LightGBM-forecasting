import numpy as np
import pandas as pd
import pytest
from src.ml.predict import predict_demand


@pytest.mark.parametrize("horizon", [7, 14])
def test_forecast_shape_dates_values(trained, horizon):
    artifact, path, _ = trained
    result = predict_demand("HOSP_A", "PGMED001", horizon, model_path=path)
    assert len(result["predictions"]) == horizon
    assert result["model_version"] == "lightgbm-v1"
    dates = pd.to_datetime([row["date"] for row in result["predictions"]])
    assert dates[0] == artifact["history"].date.max() + pd.Timedelta(days=1)
    assert (dates[1:] - dates[:-1] == pd.Timedelta(days=1)).all()
    values = [row["predicted_demand"] for row in result["predictions"]]
    assert np.isfinite(values).all() and min(values) >= 0
    assert result == predict_demand("HOSP_A", "PGMED001", horizon, model_path=path)


@pytest.mark.parametrize("hospital,medicine", [("UNKNOWN", "PGMED001"), ("HOSP_A", "UNKNOWN")])
def test_unknown_ids(trained, hospital, medicine):
    with pytest.raises(ValueError, match="Unknown hospital-medicine"):
        predict_demand(hospital, medicine, model_path=trained[1])


def test_known_ids_unknown_combination(trained):
    artifact = dict(trained[0])
    history = artifact["history"]
    artifact["history"] = history[~((history.hospital_id == "HOSP_A") & (history.medicine_id == "PGMED001"))]
    with pytest.raises(ValueError, match="Unknown hospital-medicine"):
        predict_demand("HOSP_A", "PGMED001", artifact=artifact)


@pytest.mark.parametrize("horizon", [0, 1, 8, 15])
def test_invalid_horizon(trained, horizon):
    with pytest.raises(ValueError, match="7 or 14"):
        predict_demand("HOSP_A", "PGMED001", horizon, model_path=trained[1])
