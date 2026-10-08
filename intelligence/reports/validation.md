# Executed validation

Executed in the cloud on Python 3.12.14. Commands below ran inside `intelligence/` using `/workspace/.venvs/lightgbm-forecasting/bin/python`. The PowerShell instructions are provided for Windows; PowerShell itself was not available for execution here.

- `python -m pip install -r requirements.txt`: succeeded.
- `python -m pip check`: `No broken requirements found.`
- `python -m src.ml.synthetic_data`: `Generated 3600 explicitly synthetic rows` (180 days, 4 hospitals, 5 medicines).
- `python -m src.ml.train`: succeeded, `best iteration=120`, saved `models/lightgbm_v1.joblib`.
- `python -m src.ml.evaluate`: succeeded, wrote `evaluation.json` and `prediction_vs_actual.csv`.
- `python -m pytest tests -q`: **36 passed, 1 warning in 1.32s**.
- From repository root, `python -m pytest src/test_data.py -q`: **4 passed in 0.01s**.
- `python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000`: startup completed; real HTTP checks succeeded. The smoke-test server was stopped afterward.

Actual 14-day fixed-origin recursive holdout metrics (280 pair/date observations):

| Metric | LightGBM | Seasonal-naive |
| --- | ---: | ---: |
| MAE | 5.715917592470695 | 6.896428571428571 |
| WAPE (ratio) | 0.07075093611651981 | 0.08536315812740373 |
| RMSE | 9.320308122693694 | 10.958199800019292 |

The improvement flag was `true` for this synthetic holdout only. This does not validate performance on real hospital demand.

Real HTTP results:

- `GET /health`: 200, `{"status":"ok"}`.
- `GET /model/info`: 200, version `lightgbm-v1`, origin `synthetic`.
- `POST /predict` for HOSP_A / PGMED001, horizon 7: 200, exactly 7 predictions; first date `2026-10-09`, first predicted demand `37.80077312388985`.
- Same pair, horizon 14: 200, exactly 14 predictions.
- Unknown hospital: 400, `Unknown hospital-medicine combination: UNKNOWN/PGMED001`.

TestClient also verified missing-model 503 responses and unsupported-horizon 422 responses. The training tests verified that the short observed history is rejected and that changing held-out target values does not alter either model's holdout forecasts. Repeating training produced the same selected iteration count and full-dataset holdout metrics.

There were no failing final commands or tests. Initial training emitted a LightGBM `eval_set` deprecation warning; the implementation now uses its supported `eval_X`/`eval_y` arguments. The remaining warning is Starlette's notice that its httpx TestClient integration is deprecated; the requested TestClient tests pass. No warnings or assertions were suppressed.

The original preparation scripts and observed datasets were unchanged.
