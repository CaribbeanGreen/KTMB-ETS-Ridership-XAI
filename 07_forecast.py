"""
07_forecast.py  -  forecast mode: predict FUTURE dates that have no ridership data yet
FYP: An Explainable Prediction Model Using Random Forest for KTMB ETS Ridership Surges

Why this works: the model never uses past ridership as an input. Its 13 inputs are calendar
facts (known for any date), DOSM district figures (carried forward) and weather. Only the
weather has to come from somewhere for a future date:

    dates up to today ............. Open-Meteo recent weather (forecast API, past_days)
    next 16 days .................. Open-Meteo weather FORECAST (forecast API)
    further ahead ................. that district's average for the month (2023 to 2026 archive)

Steps
  1. FORECAST MODEL: the Random Forest settings chosen by 04 (models/best_params.json), retrained on ALL data (2023 to the latest
     2026 day), so it has seen 2026's higher demand. The evaluated model (rf_final.npz,
     2023-2025 only) is NOT changed; Chapter 4 results still come from 04/05/06.
  2. FUTURE ROWS: one row per district per day, from the day after the dataset ends to --until,
     with the same calendar features as 02_build_dataset.py (its own function is reused).
  3. WEATHER: fetched from Open-Meteo as above (no API key needed). If the internet is down,
     monthly averages are used for every day and the script says so.
  4. PREDICT + EXPLAIN: Random Forest prediction and TreeSHAP for every future row.
  5. OUTPUT: data/processed/forecast/forecast.csv and dashboard/forecast.js (read by the dashboard).

Run:   python 07_forecast.py                      (about 10-20 minutes; needs internet for the weather)
       python 07_forecast.py --until 2027-01-31   (forecast further ahead)
       python 07_forecast.py --reuse              (reuse models/rf_forecast.npz, only refresh weather)
Easiest: start the dashboard with 'Start Dashboard.bat' and press 'Update forecast' (it runs this script).
A daily update only recomputes TreeSHAP for rows whose weather changed, so it takes a few minutes.
"""
import argparse
import contextlib
import importlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ets_features import FEATURES, TARGET, add_ramadan, load_dataset
from ets_rf import RandomForest
from ets_treeshap import forest_shap_cached, model_tag

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed" / "ets_district_daily.csv"
LOOKUP = BASE / "station_lookup.csv"
MODEL = BASE / "models" / "rf_forecast.npz"
OUT = BASE / "data" / "processed" / "forecast"
DASH = BASE / "dashboard"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_DAYS = 16                     # Open-Meteo forecasts at most 16 days ahead
MAX_PAST_DAYS = 92                     # ... and returns at most 92 past days
DEFAULT_PARAMS = dict(n_trees=100, max_depth=15, min_samples_leaf=3, max_features=0.5, random_state=42)

def load_params(models_dir, default):
    """Use the settings chosen by 04_train_model.py (--tune) if it saved them, otherwise the defaults."""
    f = Path(models_dir) / "best_params.json"
    if not f.exists():
        return dict(default)
    saved = json.loads(f.read_text())
    return {**default, **{k: v for k, v in saved.items() if k in default}}

DISTRICT_FEATURES = ["population_k", "hh_income_median_rm"]
TEMPORAL = [f for f in FEATURES if f not in DISTRICT_FEATURES]
WX_RECENT, WX_FORECAST, WX_AVERAGE = 0, 1, 2
WX_LABELS = ["Open-Meteo recent weather", "Open-Meteo weather forecast", "Monthly average weather (2023-2026)"]
FESTIVAL_NAME = {"hari_raya_aidilfitri": "Hari Raya Aidilfitri", "chinese_new_year": "Chinese New Year",
                 "deepavali": "Deepavali", "hari_raya_haji": "Hari Raya Haji", "christmas": "Christmas"}

build = importlib.import_module("02_build_dataset")      # reuse its calendar code, not a copy of it


def today_myt():
    return pd.Timestamp.now(tz="Asia/Kuala_Lumpur").tz_localize(None).normalize()


# ---------------------------------------------------------------------------
# future rows + calendar features
# ---------------------------------------------------------------------------
def future_calendar(districts, lookup, start, until):
    """Same calendar features as the training data, for dates that have no ridership yet.
    The calendar is built from 1 January of the start year to 120 days past the end so that
    'days since the last festival' and 'days to the next festival' are correct at both edges."""
    state_of = lookup.drop_duplicates("district").set_index("district")["state"]
    span = pd.date_range(pd.Timestamp(start.year, 1, 1), until + pd.Timedelta(days=120))
    grid = pd.DataFrame({"date": span}).merge(
        pd.DataFrame({"district": districts, "state": [state_of[d] for d in districts]}), how="cross")
    with contextlib.redirect_stdout(io.StringIO()):         # 02's progress lines are not needed here
        cal = build.add_calendar(grid, lookup)
    cal = add_ramadan(cal)
    return cal[(cal["date"] >= start) & (cal["date"] <= until)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# weather
# ---------------------------------------------------------------------------
def parse_forecast(payload):
    """Open-Meteo daily JSON -> date, temp_mean_c, rain_mm. If the mean temperature is not
    returned, it is taken as the midpoint of the daily max and min."""
    d = payload["daily"]
    w = pd.DataFrame({"date": pd.to_datetime(d["time"])})
    mean = d.get("temperature_2m_mean")
    if mean is None or all(v is None for v in mean):
        mean = [(a + b) / 2 if a is not None and b is not None else None
                for a, b in zip(d["temperature_2m_max"], d["temperature_2m_min"])]
    w["temp_mean_c"] = pd.to_numeric(pd.Series(mean), errors="coerce")
    w["rain_mm"] = pd.to_numeric(pd.Series(d["precipitation_sum"]), errors="coerce")
    return w.dropna()


def fetch_weather(points, start, today):
    """Recent + forecast weather for each district point. Returns {} if the API cannot be reached."""
    past = int(min(max((today - start).days, 0), MAX_PAST_DAYS))
    got = {}
    for i, (_, p) in enumerate(points.iterrows()):
        params = {"latitude": p["lat"], "longitude": p["lon"], "timezone": "Asia/Kuala_Lumpur",
                  "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum",
                  "past_days": past, "forecast_days": FORECAST_DAYS}
        for attempt in range(3):
            try:
                r = requests.get(FORECAST_URL, params=params, timeout=30)
                if r.status_code == 429:                    # rate limited: back off
                    time.sleep(10 * (attempt + 1))
                    continue
                r.raise_for_status()
                got[p["district"]] = parse_forecast(r.json())
                break
            except (requests.RequestException, KeyError, ValueError) as e:
                if attempt == 2:
                    print(f"   could not get weather for {p['district']}: {e}")
                time.sleep(2)
        if i == 0 and not got:                              # first call failed: assume no internet
            print("   Open-Meteo not reachable: using monthly averages for every day.")
            return {}
        time.sleep(0.3)                                     # be polite to the free API
    return got


def attach_weather(fut, history, points, start, today, offline):
    """Fill temp_mean_c / rain_mm for every future row and record where it came from."""
    clim = history.assign(m=history["date"].dt.month).groupby(["district", "m"])[["temp_mean_c", "rain_mm"]].mean()
    fut = fut.copy()
    fut["m"] = fut["date"].dt.month
    fut = fut.join(clim, on=["district", "m"])
    fut["wx"] = WX_AVERAGE
    if not offline:
        api = fetch_weather(points, start, today)
        for district, w in api.items():
            w = w.set_index("date")
            mask = (fut["district"] == district) & fut["date"].isin(w.index)
            idx = fut.loc[mask, "date"]
            fut.loc[mask, "temp_mean_c"] = w.loc[idx, "temp_mean_c"].to_numpy()
            fut.loc[mask, "rain_mm"] = w.loc[idx, "rain_mm"].to_numpy()
            fut.loc[mask, "wx"] = np.where(idx > today, WX_FORECAST, WX_RECENT)
    return fut.drop(columns="m")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", help="last date to forecast, YYYY-MM-DD (default: 31 Dec, at least 60 days ahead)")
    ap.add_argument("--reuse", action="store_true", help="reuse models/rf_forecast.npz instead of retraining")
    ap.add_argument("--offline", action="store_true", help="skip Open-Meteo; use monthly averages only")
    args = ap.parse_args()
    params = load_params(MODEL.parent, DEFAULT_PARAMS)
    for p in (OUT, DASH, MODEL.parent):
        p.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA)
    lookup = pd.read_csv(LOOKUP, dtype={"holiday_subdiv": str})
    districts = sorted(df["district"].unique())
    today = today_myt()
    start = df["date"].max() + pd.Timedelta(days=1)
    until = pd.Timestamp(args.until) if args.until else max(pd.Timestamp(today.year, 12, 31),
                                                             today + pd.Timedelta(days=60))
    print(f"Data ends {df['date'].max().date()}; forecasting {start.date()} to {until.date()} "
          f"for {len(districts)} districts (today is {today.date()}).")

    # ---- 1. forecast model on all data
    if args.reuse and MODEL.exists():
        rf = RandomForest.load(MODEL)
        print(f"1) Reusing {MODEL.name}")
    else:
        print(f"1) Training the forecast model on all {len(df):,} rows "
              f"({df['date'].min().date()} to {df['date'].max().date()}) ...")
        t0 = time.time()
        rf = RandomForest(**params).fit(df[FEATURES].to_numpy(float), df[TARGET].to_numpy(float),
                                        FEATURES, verbose=True)
        rf.save(MODEL)
        print(f"   done in {time.time() - t0:.0f}s, saved {MODEL.name}")

    # ---- 2. future rows with calendar features
    fut = future_calendar(districts, lookup, start, until)
    latest = df.sort_values("date").groupby("district").last()
    for c in DISTRICT_FEATURES:
        fut[c] = fut["district"].map(latest[c])                # latest DOSM figure carried forward
    print(f"2) {len(fut):,} future district-days built")

    # ---- 3. weather
    points = lookup.drop_duplicates("district").set_index("district").loc[districts, ["lat", "lon"]].reset_index()
    print("3) Weather from Open-Meteo ..." if not args.offline else "3) Offline: monthly average weather")
    fut = attach_weather(fut, df, points, start, today, args.offline)
    counts = fut["wx"].value_counts().reindex([WX_RECENT, WX_FORECAST, WX_AVERAGE], fill_value=0)
    for code, n in counts.items():
        print(f"   {WX_LABELS[code]}: {n:,} rows")

    # ---- 4. predict + explain
    X = fut[FEATURES].to_numpy(float)
    fut["pred"] = rf.predict(X)
    print(f"4) TreeSHAP for {len(fut):,} rows (rows unchanged since the last run are reused) ...")
    t0 = time.time()

    def progress(t, n, rows):
        if t % 10 == 0:
            done = time.time() - t0
            print(f"   tree {t}/{n} for {rows:,} new rows  ({done:.0f}s, about {done / t * (n - t):.0f}s left)", flush=True)

    phi, base, n_new = forest_shap_cached(rf, X, OUT / "shap_cache.npz", model_tag(MODEL), progress)
    gap = np.abs(base + phi.sum(axis=1) - fut["pred"].to_numpy()).max()
    print(f"   {n_new:,} rows computed, {len(fut) - n_new:,} reused, {time.time() - t0:.0f}s; "
          f"additivity check max error {gap:.1e} trips")

    # "normal day" = the model's own prediction for an ordinary day (no festival, holiday, holiday eve
    # or Ramadan) of the same district and weekday in the forecast period. Comparing the model with
    # itself keeps the surge % fair even if the overall demand level has drifted (see Chapter 4).
    # Districts/weekdays with no ordinary day in the period fall back to the last 12 months of actuals.
    ordinary = lambda t: t[(t["is_festive_period"] == 0) & (t["is_public_holiday"] == 0)
                           & (t["is_holiday_eve"] == 0) & (t["is_ramadan"] == 0)]
    norm = ordinary(fut).groupby(["district", "day_of_week"])["pred"].mean().rename("normal")
    recent = df[df["date"] > df["date"].max() - pd.Timedelta(days=365)]
    fallback = ordinary(recent).groupby(["district", "day_of_week"])[TARGET].mean().rename("normal")
    fut = fut.join(norm.combine_first(fallback), on=["district", "day_of_week"])

    # ---- 5. outputs
    cols = ["date", "district", "pred", "normal", "temp_mean_c", "rain_mm", "wx", "holiday_name", "festival"]
    out = fut[cols].copy()
    out["weather_source"] = out["wx"].map(dict(enumerate(WX_LABELS)))
    out.drop(columns="wx").round({"pred": 1, "normal": 1, "temp_mean_c": 1, "rain_mm": 1}).to_csv(OUT / "forecast.csv", index=False)

    ti = [FEATURES.index(f) for f in TEMPORAL]
    di = [FEATURES.index(f) for f in DISTRICT_FEATURES]
    rows = []
    for k, r in enumerate(fut.itertuples(index=False)):
        rows.append([
            r.date.strftime("%Y-%m-%d"), r.district, round(float(r.pred), 1), round(float(r.normal), 1),
            round(base + float(phi[k, di].sum()), 1),
            [round(float(v), 1) for v in phi[k, ti]],
            [int(r.day_of_week), int(r.month), int(r.is_public_holiday), int(r.is_holiday_eve),
             int(r.days_to_festival), int(r.days_since_festival), int(r.is_festive_period),
             int(r.is_ramadan), round(float(r.temp_mean_c), 1), round(float(r.rain_mm), 1), int(r.is_weekend)],
            r.holiday_name if isinstance(r.holiday_name, str) else "",
            FESTIVAL_NAME.get(r.festival, "") if isinstance(r.festival, str) else "",
            int(r.wx),
        ])
    payload = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "issued": today.strftime("%Y-%m-%d"),
        "period": [start.strftime("%Y-%m-%d"), until.strftime("%Y-%m-%d")],
        "trainedOn": [str(df["date"].min().date()), str(df["date"].max().date())],
        "trainRows": int(len(df)),
        "baseValue": round(base, 1),
        "weatherLabels": WX_LABELS,
        "rowFields": ["date", "districtName", "pred", "normalSameWeekday", "districtLevel", "shap", "values",
                      "holiday", "festival", "wx"],
        "rows": rows,
    }
    (DASH / "forecast.js").write_text("window.ETS_FORECAST = " + json.dumps(payload, separators=(",", ":")) + ";\n",
                                      encoding="utf-8")

    net = fut.groupby("date")["pred"].sum()
    print(f"\n5) Wrote {OUT / 'forecast.csv'} and {DASH / 'forecast.js'}")
    print("Busiest forecast days for the whole network:")
    for d, v in net.sort_values(ascending=False).head(5).items():
        fest = fut.loc[fut["date"] == d, "festival"].iloc[0]
        print(f"   {d.date()}  {v:,.0f} trips" + (f"  ({FESTIVAL_NAME.get(fest, fest)})" if fest else ""))
    print(f"Open {DASH / 'index.html'} in your browser.")


if __name__ == "__main__":
    main()
