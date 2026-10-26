"""
06_prepare_dashboard.py  -  final model + 2026 holdout predictions + explanations for the dashboard
FYP: An Explainable Prediction Model Using Random Forest for KTMB ETS Ridership Surges

Steps
  1. FINAL MODEL: train the Random Forest on ALL study data (2023-2025), settings chosen by
     04_train_model.py (models/best_params.json), and save it to models/rf_final.npz
  2. HOLDOUT TEST: predict every 2026 district-day (data the model has never seen) and score it
     against the actual ridership and against the naive baseline -> data/processed/holdout/
  3. EXPLANATIONS: TreeSHAP for every 2026 row (about 5 minutes)
  4. DASHBOARD DATA: write dashboard/data.js, read by dashboard/index.html

Run:   python 06_prepare_dashboard.py              (about 15-40 minutes with the tuned model)
       python 06_prepare_dashboard.py --reuse      (reuse models/rf_final.npz; TreeSHAP only for new rows,
                                                    so after a data update it takes minutes, not 40)
Then open dashboard/index.html in a browser.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ets_features import FEATURES, TARGET, load_dataset
from ets_rf import RandomForest, mae, mape, r2, rmse
from ets_treeshap import forest_shap_cached, model_tag

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed" / "ets_district_daily.csv"
MODEL = BASE / "models" / "rf_final.npz"
HOLD = BASE / "data" / "processed" / "holdout"
DASH = BASE / "dashboard"

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
FESTIVAL_NAME = {"hari_raya_aidilfitri": "Hari Raya Aidilfitri", "chinese_new_year": "Chinese New Year",
                 "deepavali": "Deepavali", "hari_raya_haji": "Hari Raya Haji", "christmas": "Christmas"}


def score(df, col):
    y, p = df[TARGET], df[col]
    net = df.groupby("date")[[TARGET, col]].sum()
    fest = df[df["is_festive_period"] == 1].groupby("date")[[TARGET, col]].sum()
    return {"row_r2": r2(y, p), "row_mae": mae(y, p), "row_rmse": rmse(y, p),
            "network_r2": r2(net[TARGET], net[col]), "network_mae": mae(net[TARGET], net[col]),
            "network_mape": mape(net[TARGET], net[col]),
            "festive_network_mape": mape(fest[TARGET], fest[col]) if len(fest) else None,
            "festive_days": int(len(fest))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="reuse models/rf_final.npz instead of retraining")
    args = ap.parse_args()
    params = load_params(MODEL.parent, DEFAULT_PARAMS)
    for p in (HOLD, DASH, MODEL.parent):
        p.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA)
    study = df[df["split"] == "study"]
    hold = df[df["split"] == "holdout"].copy()

    # ---- 1. final model on all study years
    if args.reuse and MODEL.exists():
        rf = RandomForest.load(MODEL)
        print(f"Reusing {MODEL.name}")
    else:
        print(f"1) Training the final model on {len(study):,} rows (2023-2025) ...")
        t0 = time.time()
        rf = RandomForest(**params).fit(study[FEATURES].to_numpy(float), study[TARGET].to_numpy(float),
                                        FEATURES, verbose=True)
        rf.save(MODEL)
        print(f"   done in {time.time() - t0:.0f}s, saved {MODEL.name}")

    # ---- 2. holdout test on 2026
    X = hold[FEATURES].to_numpy(float)
    hold["pred"] = rf.predict(X)
    normal = study[(study["is_festive_period"] == 0) & (study["is_public_holiday"] == 0)
                   & (study["is_holiday_eve"] == 0) & (study["is_ramadan"] == 0)]
    weekday_norm = normal.groupby(["district", "day_of_week"])[TARGET].mean().rename("normal")
    hold = hold.merge(weekday_norm, on=["district", "day_of_week"], how="left")
    naive = study.groupby(["district", "day_of_week"])[TARGET].mean().rename("naive")
    hold = hold.merge(naive, on=["district", "day_of_week"], how="left")
    metrics = {"period": [str(hold["date"].min().date()), str(hold["date"].max().date())],
               "rows": len(hold), "rf": score(hold, "pred"), "naive_baseline": score(hold, "naive")}
    # month-by-month bias: shows whether 2026 demand moved outside the range seen in training
    net = hold.groupby("date")[[TARGET, "pred"]].sum()
    month = net.groupby(net.index.to_period("M")).sum()
    bias = ((month["pred"] / month[TARGET] - 1) * 100).round(1)
    metrics["monthly_bias_pct"] = {str(k): float(v) for k, v in bias.items()}
    (HOLD / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2))
    hold[["date", "district", TARGET, "pred", "naive", "is_festive_period"]].to_csv(
        HOLD / "holdout_predictions.csv", index=False)
    r, n = metrics["rf"], metrics["naive_baseline"]
    print(f"\n2) HOLDOUT TEST, {metrics['period'][0]} to {metrics['period'][1]} (never seen in training)")
    print(f"{'':26s}{'Random Forest':>15s}{'Naive':>10s}")
    for k, lab in [("row_r2", "Row R2"), ("network_r2", "Network daily R2"),
                   ("network_mape", "Network daily MAPE %"), ("festive_network_mape", "Festive-day MAPE %")]:
        print(f"{lab:26s}{r[k]:>15.3f}{n[k]:>10.3f}")
    print("Network prediction bias by month (negative = under-predicting):")
    print("   " + ", ".join(f"{k[5:]}: {v:+.1f}%" for k, v in metrics["monthly_bias_pct"].items()))

    # ---- 3. TreeSHAP for every holdout row (cached: only new or changed rows are computed)
    cache = HOLD / "shap_holdout.npz"
    tag = model_tag(MODEL)
    if cache.exists():                       # older format (phi only, same rows): convert it once
        z = np.load(cache, allow_pickle=False)
        if "X" not in z.files and args.reuse and z["phi"].shape == X.shape:
            np.savez_compressed(cache, tag=np.array(tag), X=X, phi=z["phi"])
    print(f"\n3) TreeSHAP explanations for {len(hold):,} rows ({len(rf.trees)} trees) ...")
    t0 = time.time()

    def progress(t, n, rows):
        if t % 10 == 0:
            done = time.time() - t0
            print(f"   tree {t}/{n} for {rows:,} new rows  ({done:.0f}s, about {done / t * (n - t):.0f}s left)", flush=True)

    phi, base, n_new = forest_shap_cached(rf, X, cache, tag, progress)
    print(f"   {n_new:,} rows computed, {len(hold) - n_new:,} reused from the cache ({time.time() - t0:.0f}s)")
    gap = np.abs(base + phi.sum(axis=1) - hold["pred"].to_numpy()).max()
    print(f"   additivity check (base + SHAP = prediction): max error {gap:.1e} trips")

    # ---- 4. dashboard data
    districts = sorted(hold["district"].unique())
    latest = hold.sort_values("date").groupby("district").last()
    states = latest["state"].to_dict()
    study_dates = np.sort(study["date"].unique())
    cut = pd.Timestamp(study_dates[int(len(study_dates) * 0.8)])
    ti = [FEATURES.index(f) for f in TEMPORAL]
    di = [FEATURES.index(f) for f in DISTRICT_FEATURES]
    rows = []
    for k, (_, r_) in enumerate(hold.iterrows()):
        rows.append([
            r_["date"].strftime("%Y-%m-%d"), districts.index(r_["district"]), int(r_[TARGET]),
            round(float(r_["pred"]), 1), round(float(r_["normal"]), 1),
            round(base + float(phi[k, di].sum()), 1),                  # normal level for this district
            [round(float(v), 1) for v in phi[k, ti]],                  # calendar & weather contributions
            [int(r_["day_of_week"]), int(r_["month"]), int(r_["is_public_holiday"]), int(r_["is_holiday_eve"]),
             int(r_["days_to_festival"]), int(r_["days_since_festival"]), int(r_["is_festive_period"]),
             int(r_["is_ramadan"]), round(float(r_["temp_mean_c"]), 1), round(float(r_["rain_mm"]), 1),
             int(r_["is_weekend"])],
            r_["holiday_name"] if isinstance(r_["holiday_name"], str) else "",
            FESTIVAL_NAME.get(r_["festival"], "") if isinstance(r_["festival"], str) else "",
        ])
    payload = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "districts": [{"name": d, "state": states[d], "population_k": round(float(latest.loc[d, "population_k"]), 1),
                       "income": int(latest.loc[d, "hh_income_median_rm"])} for d in districts],
        "dataset": {
            "sources": ["KTMB ETS ridership (data.gov.my)", "Open-Meteo historical weather",
                        "DOSM population & household income", "Federal & state public holidays"],
            "studyRows": int(len(study)), "studyPeriod": [str(study["date"].min().date()), str(study["date"].max().date())],
            "trainPeriod": [str(study["date"].min().date()), str((cut - pd.Timedelta(days=1)).date())],
            "testPeriod": [str(cut.date()), str(study["date"].max().date())],
            "holdoutRows": int(len(hold)), "districts": len(districts), "features": len(FEATURES),
            "trees": len(rf.trees),
            "params": {"n_trees": len(rf.trees), "max_depth": rf.max_depth, "min_samples_leaf": rf.min_samples_leaf,
                       "max_features": rf.max_features},
            "tuned": bool(json.loads((MODEL.parent / "best_params.json").read_text()).get("tuned"))
                     if (MODEL.parent / "best_params.json").exists() else False,
            "cvFolds": 3},
        "temporal": TEMPORAL,
        "rowFields": ["date", "district", "actual", "pred", "normalSameWeekday", "districtLevel", "shap",
                      "values", "holiday", "festival"],
        "valueFields": ["day_of_week", "month", "is_public_holiday", "is_holiday_eve", "days_to_festival",
                        "days_since_festival", "is_festive_period", "is_ramadan", "temp_mean_c", "rain_mm",
                        "is_weekend"],
        "baseValue": round(base, 1),
        "metrics": {"network_r2": round(r["network_r2"], 3), "network_mape": round(r["network_mape"], 2),
                    "festive_mape": round(r["festive_network_mape"], 2),
                    "naive_festive_mape": round(n["festive_network_mape"], 2)},
        "rows": rows,
    }
    (DASH / "data.js").write_text("window.ETS_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
                                  encoding="utf-8")
    print(f"\n4) Wrote {DASH / 'data.js'} ({(DASH / 'data.js').stat().st_size / 1e6:.1f} MB)")
    print(f"Open {DASH / 'index.html'} in your browser.")


if __name__ == "__main__":
    main()
