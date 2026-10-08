# PulseGrid AI V1 — Implementation and Mentor Walkthrough

## 1. What we built

PulseGrid AI V1 predicts **daily requested medicine demand** for a hospital and medicine pair over the next **7 or 14 days**. It includes a pooled LightGBM model, a seasonal-naive comparison model, chronological evaluation, a persisted model artifact, automated tests, and a standalone FastAPI service.

The implementation lives in `intelligence/`. It does not connect to Supabase, change the main backend, call an LLM, or calculate shortages or redistribution plans. Those are later modules that can consume the forecasts.

**A short explanation to give your mentor:**

> “We built the first demand forecasting layer of PulseGrid AI. It learns patterns across hospital-medicine combinations using past demand, calendar information, and categorical IDs. We evaluate it against a simple weekly baseline using future dates that the model has not seen. The API loads a saved model and returns seven or fourteen daily forecasts. Our current results are from clearly labeled synthetic data because the available observed history is too short for a reliable evaluation.”

## 2. Repository and environment work

1. Initially inspected the repository, which contained only a README and license, and prepared an isolated Python 3.12 environment for LightGBM development.
2. When the repository gained the preparation code and datasets, fetched the new `main` commit and fast-forwarded the clean checkout. No existing preparation scripts or observed datasets were overwritten.
3. Installed and validated the forecasting dependencies, then added the standalone V1 module.
4. Generated synthetic training data, trained and evaluated the model, ran the new and existing tests, and exercised the API through real HTTP requests.
5. Added Windows PowerShell instructions, module documentation, saved evaluation reports, and this mentor walkthrough.

The cloud virtual environment and temporary helper scripts are machine-local. Repository requirements and documentation describe how to reproduce the application on another machine. Publication of a cloud environment is a separate product action; this report concerns the code and validated implementation.

## 3. Why synthetic data was necessary

The observed file is `data/processed/demand_history_clean.csv`. It contains **300 observations**, spanning **September 24–October 8, 2026**, across **4 hospitals × 5 medicines**.

Fifteen days are not enough for fourteen historical warm-up days plus meaningful training, validation, and a seven/fourteen-day holdout. The loader validates this dataset, but training on it for the required evaluation raises an explicit insufficient-history error.

We generated **180 days × 20 hospital-medicine pairs = 3,600 observations**, from **April 12–October 8, 2026**, in `data/processed/demand_history_synthetic.csv`. Each row has `data_origin=synthetic`. The generator uses seed **42** and simulates:

- Hospital-specific and medicine-specific demand levels.
- Weekly seasonality and gradual trends.
- Random variation.
- Occasional multi-day outbreak-related demand increases.

Canonical hospital/medicine IDs and medicine names come from the existing data. The generator refuses to overwrite an existing output or the observed demand file. It does not create additional real hospital evidence.

**Tell the mentor:** synthetic data demonstrates that the pipeline works; it does not demonstrate clinical or operational accuracy.

## 4. Data correctness

`data_loader.py` reads CSVs and rejects inconsistencies rather than silently repairing them. It checks required columns, valid daily dates, nonempty IDs, finite nonnegative requested demand, unique `(date, hospital_id, medicine_id)` keys, and missing daily dates inside each series. It returns observations in chronological order.

The target is **quantity_requested**, not quantity_dispensed. Canonical **medicine_id** defines the series; source_medicine_id is never used as a join key. Training additionally requires all series to share a common daily date range so splits are comparable.

## 5. How the model works

A **pooled model** is one model trained across all hospital-medicine series, rather than a separate model for each pair. IDs are categorical, allowing it to learn differences while sharing patterns across series.

The model uses exactly eight features:

| Feature | Meaning |
| --- | --- |
| hospital_id | Which hospital; categorical |
| medicine_id | Which canonical medicine; categorical |
| day_of_week | Calendar day, Monday–Sunday |
| month | Calendar month |
| lag_1 | Demand one day earlier |
| lag_7 | Demand seven days earlier |
| rolling_mean_7 | Average demand over the previous seven days |
| rolling_mean_14 | Average demand over the previous fourteen days |

Rolling means shift the target before calculating the window. The feature function is shared by training and inference, and category mappings plus feature order are saved with the model.

No stock-out warnings, future stock, risk labels, quantity dispensed, or outbreak flags are used as predictors. Simulated outbreak flags exist in the dataset but future outbreak information is not assumed to be available to V1.

LightGBM combines multiple small decision trees. We use conservative settings: at most 15 leaves, depth 5, minimum 30 samples per leaf, learning rate 0.04, L2 regularization 2.0, a fixed seed, and two training threads. Early stopping selected **120 trees** out of a maximum of 250.

## 6. Evaluation and leakage prevention

There is no random train/test split. Every series uses the same boundaries:

| Stage | Dates | Purpose |
| --- | --- | --- |
| Training history | April 12–September 10, 2026 | Initial fitting; first 14 days are feature warm-up |
| Validation | September 11–24, 2026 | Choose the number of trees with early stopping |
| Final test | September 25–October 8, 2026 | Held-out recursive evaluation |

Usable initial training features begin April 26. Validation is chronological **one-step** evaluation: earlier observed validation demand can form features for later validation dates. After tree-count selection, the evaluation model is refitted on training plus validation observations through September 24.

The final test starts from each pair's last available pre-test history. For day one, the model predicts demand. That prediction is appended to history, then used to predict day two, and so on. **Actual test demand is never supplied to this recursive loop.** Actual values are read afterward only for scoring.

The baseline repeats the most recent seven pre-test daily demand values. Both models are compared on exactly the same hospital/medicine/date rows. Reports include the first seven forecast days and all fourteen forecast days from the same origin; these are not independent backtests.

After evaluation, a separate serving fit uses the full dataset through October 8 with the selected tree count. Its saved history supports predictions starting October 9. The report explicitly distinguishes serving training dates from evaluation training dates; the metrics belong to the pre-test model.

## 7. Actual measured results

The full 14-day holdout contains **280 predictions**: 20 pairs × 14 days.

| Metric | LightGBM | Seasonal-naive baseline |
| --- | ---: | ---: |
| MAE | 5.715918 | 6.896429 |
| WAPE | 7.075094% | 8.536316% |
| RMSE | 9.320308 | 10.958200 |

For the first 7 forecast days, covering 140 predictions:

| Metric | LightGBM | Seasonal-naive baseline |
| --- | ---: | ---: |
| MAE | 6.784770 | 7.671429 |
| WAPE | 8.331443% | 9.420226% |
| RMSE | 10.682693 | 12.143781 |

- **MAE** is average absolute error in daily demand units.
- **WAPE** is total absolute error divided by total actual demand. JSON stores a ratio, not a percentage. For zero actual-demand totals, it returns zero for an exact forecast and `null` for nonzero error.
- **RMSE** gives larger errors more weight and is in demand units.

The report's improvement flag requires lower MAE and no worse WAPE/RMSE. LightGBM met that rule on this synthetic holdout, with about **17.1% lower 14-day MAE** than the baseline. These numbers were computed by the executed pipeline, not invented.

Do not present WAPE as “model accuracy,” and do not claim these results apply to real hospital demand. There is one evaluation origin and no uncertainty interval.

## 8. Files and why they exist

| File | Role |
| --- | --- |
| `intelligence/src/ml/data_loader.py` | Reject bad data before it reaches the model |
| `intelligence/src/ml/synthetic_data.py` | Produce a reproducible, explicitly simulated development dataset |
| `intelligence/src/ml/features.py` | Keep causal feature construction and category definitions consistent |
| `intelligence/src/ml/baseline.py` | Provide an honest simple forecasting benchmark |
| `intelligence/src/ml/train.py` | Split chronologically, select tree count, evaluate, refit and save artifacts |
| `intelligence/src/ml/evaluate.py` | Calculate metrics and export JSON/CSV without retraining |
| `intelligence/src/ml/predict.py` | Load the artifact, validate pairs, and forecast recursively |
| `intelligence/src/api/main.py` | Expose typed standalone FastAPI endpoints |
| `intelligence/tests/` | Verify data errors, leakage prevention, training, predictions and API behavior |
| `intelligence/requirements.txt` | Pin direct dependencies validated with Python 3.12 |
| `intelligence/README.md` | Provide detailed commands, architecture and API instructions |
| `intelligence/models/lightgbm_v1.joblib` | Save the trained demo model and inference metadata |
| `intelligence/reports/evaluation.json` | Record origins, date splits, counts, hyperparameters and actual metrics |
| `intelligence/reports/prediction_vs_actual.csv` | Make every held-out prediction inspectable beside its actual and baseline |
| `intelligence/reports/validation.md` | Record executed commands, HTTP results and warnings |

The artifact contains the estimator, model version, ordered features, category mappings, serving training range, latest fourteen historical days for each pair, evaluation metadata and held-out results. The small trained demo artifact is included in the repository for convenience. Only load trusted joblib files: pickle-based files can execute code. Training can regenerate it; dependency/platform differences may affect binary compatibility or exact results.

## 9. Tests and actual API behavior

**36 new pytest tests passed**, and **4 pre-existing preparation-data tests passed**. Coverage includes missing/invalid columns, duplicate records, negative/nonfinite demand, missing dates, feature leakage, artifact creation, aligned chronological splits, seven/fourteen forecast counts, finite nonnegative outputs, unknown IDs and combinations, API validation, and missing-model errors.

One regression test changes held-out target demand to a very large value and verifies that neither model's evaluation forecasts nor the selected tree count change. This directly checks that future actual values do not influence final test predictions.

The running Uvicorn service was checked through real HTTP requests:

| Request | Observed result |
| --- | --- |
| GET `/health` | 200; `{"status":"ok"}` |
| GET `/model/info` | 200; model `lightgbm-v1`, origin `synthetic` |
| POST `/predict`, horizon 7 | 200; exactly 7 forecasts starting October 9, 2026 |
| POST `/predict`, horizon 14 | 200; exactly 14 forecasts |
| Unknown hospital | 400 with a clear unknown-combination error |

TestClient additionally verified 503 when the model is missing and 422 for an unsupported horizon. The health endpoint indicates service liveness, not that a model is trained.

There were no failing final tests. One upstream Starlette warning remains: its httpx TestClient integration is deprecated. Tests still pass; assertions and warnings were not disabled. A LightGBM deprecation warning encountered initially was resolved by using its supported validation arguments.

The server was stopped after the smoke test. Start it for your presentation. The API caches the artifact and does not train during requests; restart the service after replacing the artifact.

## 10. Windows PowerShell demo commands

From the repository root:

```powershell
cd intelligence
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# The synthetic CSV is included. Generate only if it is missing:
if (-not (Test-Path '..\data\processed\demand_history_synthetic.csv')) { python -m src.ml.synthetic_data }
python -m src.ml.train
python -m src.ml.evaluate
python -m pytest tests -q
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

If activation is blocked, run the same commands with `.\.venv\Scripts\python.exe` instead of `python`; no system policy change is required. Linux/cloud validation succeeded; these PowerShell commands were written for Windows but not executed on Windows here.

In another PowerShell window:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health'
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/model/info' | ConvertTo-Json -Depth 12
$body = @{ hospital_id = 'HOSP_A'; medicine_id = 'PGMED001'; horizon_days = 7 } | ConvertTo-Json
$result = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/predict' -ContentType 'application/json' -Body $body
$result | ConvertTo-Json -Depth 6
$result.predictions.Count
```

Alternatively, open the local Swagger page at `http://127.0.0.1:8000/docs`, select POST `/predict`, choose **Try it out**, and submit:

```json
{"hospital_id":"HOSP_A","medicine_id":"PGMED001","horizon_days":7}
```

The actual cloud smoke response's first prediction was `37.80077312388985` for `2026-10-09`; all seven predictions were calculated by the trained model. This is an example result, not a hardcoded response.

## 11. Three-minute mentor demonstration

1. **Problem:** hospitals need a forward view of requested medicine demand to support later inventory decisions.
2. **Implementation:** show the eight causal features and explain the pooled LightGBM model.
3. **Honest comparison:** show the baseline and chronological holdout report; say clearly that the data is synthetic.
4. **Working product:** submit a seven-day API request, then change the horizon to fourteen. Show dates, counts and nonnegative values.
5. **Reliability:** show the tests and an unknown-ID error; explain that missing history is rejected rather than fabricated.
6. **Next step:** gather longer observed histories, conduct rolling-origin real-data backtests, and integrate forecasts into the teammate's backend and later inventory simulator.

## 12. Likely mentor questions

**Why LightGBM?** It works well with tabular lag/calendar features, trains quickly, and supports categorical IDs. The baseline tests whether that extra complexity helps on the evaluated data.

**Why not train on the observed CSV alone?** Fifteen days do not support the requested warm-up and defensible chronological evaluation. We chose a transparent simulation instead of an unsupported accuracy claim.

**How did you prevent future-data leakage?** Shifted group-specific rolling features, chronological splits, shared category/feature definitions, and recursive testing that appends only predictions. A dedicated test checks independence from held-out target values.

**Can the system forecast a new hospital or medicine?** V1 rejects unknown pairs because it has no saved history for them. A future cold-start strategy would need explicit design and evaluation.

**Can these forecasts directly trigger transfers?** No. Forecasts alone do not establish inventory shortages, expiry constraints, or valid MOUs. Those checks belong to later simulator/optimizer modules.

**What is missing for production?** Longer representative observed data, multiple rolling-origin tests, drift monitoring, uncertainty estimation, deployment controls, and review of downstream medical supply decisions.
