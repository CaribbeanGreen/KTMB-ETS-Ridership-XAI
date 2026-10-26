# FYP Phase 2: Data Collection

KTMB ETS ridership surge dataset (Random Forest + TreeSHAP FYP).
Builds **one unified table: one row per departure district per day**, 2023–2025 (study) + 2026 (holdout).

## Run (about 5–10 minutes, needs internet)

```
pip install -U pandas pyarrow requests holidays
python 01_collect.py        # downloads raw files into data/raw/ (safe to re-run)
python 02_build_dataset.py  # builds data/processed/ets_district_daily.csv
python 03_eda.py            # EDA charts + eda_summary.txt into data/processed/eda/ (Chapter 4.3)
python 04_train_model.py --tune  # tune trees/depth/leaf with 3-fold time-series CV, then train + test (about 30 min)
python 05_explain.py        # TreeSHAP: global + local explanations, additivity / exact-enumeration / sanity checks (about 2-3 min)
python 06_prepare_dashboard.py  # final model on 2023-2025, 2026 holdout test, TreeSHAP for every 2026 row (about 15-40 min)
python 07_forecast.py       # forecast mode: future dates to 31 Dec, Open-Meteo weather forecast (about 10-20 min)
# then double-click 'Start Dashboard.bat' (dashboard with update buttons)
```

Then open `data/processed/data_quality_report.txt` and check the three things below.

## What to check after running

1. **Unmapped stations.** If 2023–2025 used a station name that 2026 doesn't (a renamed or closed station), the build stops and lists it. Add a row to `station_lookup.csv` and re-run.
2. **DOSM matching.** The report prints "matched X% of rows" for population and income. Anything under 100% names the district that failed; fix the spelling in `station_lookup.csv`. Watch for **Muallim** (split from Batang Padang in 2016; older DOSM tables may not have it).
3. **Gap days.** Dates where the whole ETS network recorded zero trips are treated as KITS data gaps and dropped, not counted as zero demand. The report lists them.

## Sources (for Chapter 3 / 4 data table)

| # | Data | Source | Level | Years |
|---|------|--------|-------|-------|
| 1 | ETS hourly origin-destination ridership | data.gov.my `ridership_od_ets` (`storage.data.gov.my/transportation/ktmb/ets_{year}.parquet`) | station, hourly | 2023–2026 |
| 2 | Population by district | DOSM `population_district` | district, annual | 2020–2024 |
| 3 | Household income (mean, median) | DOSM `hh_income_district` (HIES) | district, survey years | 2019–2024 |
| 4 | Daily temperature and rainfall | Open-Meteo Historical Weather API | point per district | 2023–2026 |
| 5 | Federal + state public holidays | `holidays` Python package (data only, no modelling) | state, daily | 2023–2026 |

All are open data (CC BY 4.0). No personal data: ridership is already aggregated to station-hour by KTMB.

## Design decisions (write these into Chapter 3)

- **Unit of analysis:** departure district × day, as in the proposal. Hourly O-D rows are summed per origin station, then per district.
- **Station → district:** `station_lookup.csv` (50 ETS stations → 28 DOSM districts, 21 kept after the coverage rules). Pulau Sebang (Tampin) station is physically in **Alor Gajah, Melaka**, not Tampin, Negeri Sembilan.
- **Weather point:** coordinates of the district's main ETS station.
- **DOSM years:** each row takes the latest DOSM figure from a year ≤ its own year. 2025–2026 reuse 2024 population. Income is only surveyed some years, so it is carried forward to later years.
- **Festive features:** Hari Raya Aidilfitri, Hari Raya Haji, Chinese New Year, Deepavali and Christmas. `days_to_festival` / `days_since_festival` (capped at 30) and `is_festive_period` (±7 days) capture the build-up and return waves, not just the holiday itself.
- **2022 excluded:** post-pandemic recovery distorts demand.
- **2026 = holdout:** never used for training or tuning; a final real-world check only.

## Target definition

`ridership` = ETS trips departing the district **to stations on the established line** (Padang Besar to Gemas).
Trips to stations that opened during the data period (Gemas-JB extension, Klang, Rembau) are excluded:
they were 2.0% of 2025 trips but 18.6% of 2026 trips, so counting them would make 2026 look like a sudden
surge everywhere (e.g. KL -> JB Sentral trips depart from KL).

## Coverage rules (applied automatically in step 5 of the build)

- **Pre-opening rows removed.** A district with no ETS service can't have demand, so days before its first recorded departure are dropped (6,438 district-days), not kept as "zero".
- **Southern extension excluded** (`EXCLUDE_NEW_LINE = True`). Gemas–JB opened in stages: Segamat 15 Mar 2025, Kluang 30 Aug 2025, Kulai and JB Sentral 12 Dec 2025. Under a year of history can't teach festive patterns, and JB's 2026 volume (~2,000 trips a day) is far beyond anything seen in training. Discuss it as a limitation / future work.
- **Thin or unserved districts dropped:** Klang and Rembau (no service until 2026), and Ulu Selangor (6.7 trips a day).
- **Last ~6 days of 2026 dropped:** the Open-Meteo archive lags about a week.
- Result: **21 districts**, 23,016 study rows (2023–2025) + 5,586 holdout rows (2026), no missing values.

## Using the dashboard (no terminal needed after the first setup)

Double-click **Start Dashboard.bat**. It starts a small local server (`serve.py`, Python standard library
only, listening on this computer only) and opens http://localhost:8765/. Keep its window open while you use it.

- **Update forecast** (a few minutes): runs `07_forecast.py --reuse` for today's 16-day weather forecast.
  The dashboard asks for this whenever the forecast was issued before today.
- **Full data update** (30-60 minutes, about once a month): runs `01_collect.py --latest`,
  `02_build_dataset.py`, `06_prepare_dashboard.py --reuse` and `07_forecast.py` to add newly published ETS
  ridership days and retrain the forecast model. The evaluated 2023-2025 model is not retrained.
- TreeSHAP values are cached: only rows whose inputs changed are recomputed (`shap_holdout.npz`,
  `forecast/shap_cache.npz`). Progress and the full log appear on the page (`data/processed/update_log.txt`).
- Opening `index.html` directly still works, but the update buttons are then switched off.

## Hyperparameter tuning (04_train_model.py --tune, report 3.5.4)

- Grid: number of trees {50, 100, 150} x max depth {10, 15, 20} x min samples per leaf {1, 3, 5}; max_features fixed at 0.5.
- **3-fold time-series cross-validation inside the training period** (rolling origin): each fold trains on all dates
  before a ~4-month validation block, so the model is never validated on dates earlier than its training data.
- Selection: lowest mean validation network MAPE. Results: `data/processed/model/tuning_results.csv`, `tuning_folds.csv`, `fig_tuning.png`.
- The chosen setting is saved to `models/best_params.json` and reused by 06 and 07.

## Forecast mode (07_forecast.py)

The model never uses past ridership as an input, so it can predict any date once that date's weather is known:

| Dates | Weather used |
|---|---|
| up to today (after the dataset ends) | Open-Meteo recent weather (forecast API, `past_days`) |
| next 16 days | Open-Meteo weather forecast |
| further ahead | the district's average for that month, 2023-2026 archive |

- Uses a separate **forecast model** (`models/rf_forecast.npz`): same settings, retrained on all data to the latest 2026 day.
  The evaluated model (`rf_final.npz`, 2023-2025) and all Chapter 4 results are unchanged.
- "Normal day" for the surge % = the forecast model's own prediction for an ordinary day of the same district and weekday.
- Re-run `python 07_forecast.py` whenever you want fresher weather; `--until 2027-01-31` forecasts further, `--offline` skips the API.
- The dashboard's **Recommended action** (report Algorithm 3.2) is rule-based: surge level + the strongest TreeSHAP factors + heavy-rain alert.

## Known limitations

- **No school-holiday flag yet.** MOE school-holiday dates change yearly and group by state, so they have to be typed in by hand from the official MOE calendars. Worth adding: school breaks are a real surge driver.
- **Trips ≠ passengers.** data.gov.my notes ridership counts trips.

## Files

```
ets_rf.py                  Random Forest implemented with NumPy (tree, bootstrap, bagging, importance, save/load, metrics)
ets_features.py            the model's input features (shared by training, TreeSHAP and the dashboard)
04_train_model.py          train / test / tune the Random Forest
ets_treeshap.py            TreeSHAP (path-dependent, vectorised) + exact Shapley enumeration for verification
05_explain.py              explanation outputs for Chapter 4.7 -> data/processed/shap/
06_prepare_dashboard.py    final model (models/rf_final.npz), holdout results (data/processed/holdout/), dashboard/data.js
07_forecast.py             forecast model (models/rf_forecast.npz), future predictions (data/processed/forecast/), dashboard/forecast.js
dashboard/                 prediction dashboard: index.html + style.css + app.js (hand-written, no framework)
models/rf_model.npz        trained model
data/processed/model/      test results, predictions and charts
01_collect.py              download raw data
02_build_dataset.py        clean, map, merge, add features, quality report
station_lookup.csv         station -> state, district, holiday code, lat, lon
data/raw/                  untouched downloads
data/processed/
  ets_district_daily.csv   final dataset
  data_dictionary.csv      column meanings + source (paste into report appendix)
  data_quality_report.txt  counts, gaps, matching, missing values
```
