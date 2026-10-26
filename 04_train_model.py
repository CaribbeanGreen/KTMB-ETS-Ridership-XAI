"""
04_train_model.py  -  Phase 3: Random Forest model development (Chapter 4.4)
FYP: An Explainable Prediction Model Using Random Forest for KTMB ETS Ridership Surges

Trains the NumPy Random Forest (ets_rf.py) on the unified dataset and tests it.

Split (chronological, never random - this is time-series data):
  study years 2023-2025 -> first 80% of dates = TRAIN, last 20% = TEST
  2026 holdout          -> not touched here (used once, in the evaluation step)

Usage
  python 04_train_model.py            train with the default settings, test, save the model
  python 04_train_model.py --tune     first tune the number of trees, max depth and min samples per leaf
                                      with time-series cross-validation inside TRAIN (report 3.5.4),
                                      then train with the best setting (about 30 minutes)
Outputs
  models/rf_model.npz                         the trained forest (plain NumPy arrays, no pickle)
  models/best_params.json                     the settings used (read by 06 and 07)
  data/processed/model/metrics.json           all test results
  data/processed/model/test_predictions.csv   every test row: actual, predicted, naive baseline
  data/processed/model/district_errors.csv    error per district
  data/processed/model/tuning_results.csv     (with --tune) every setting, mean and per-fold scores
  data/processed/model/tuning_folds.csv       (with --tune) the cross-validation folds
  data/processed/model/fig_tuning.png         (with --tune) validation error for every setting
  data/processed/model/fig_importance.png     feature importance
  data/processed/model/fig_test_network.png   actual vs predicted, network total per day
  data/processed/model/fig_district_error.png error per district
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ets_features import FEATURES, LABELS, TARGET, load_dataset
from ets_rf import RandomForest, mae, mape, r2, rmse

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed" / "ets_district_daily.csv"
OUT = BASE / "data" / "processed" / "model"
MODELS = BASE / "models"

TRAIN_SHARE = 0.80
DEFAULT_PARAMS = dict(n_trees=100, max_depth=15, min_samples_leaf=3, max_features=0.5, random_state=42)
# Hyperparameters tuned (report 3.5.4): number of trees, maximum depth, minimum samples per leaf.
# max_features stays at 0.5 (half the features tried at each split, the usual regression default).
TUNE_GRID = dict(max_depth=[10, 15, 20], min_samples_leaf=[1, 3, 5])
TUNE_TREES = [50, 100, 150]
# Time-series cross-validation (rolling origin): the training period is cut into blocks of dates;
# each fold trains on everything BEFORE its validation block and validates on that block, so the
# model is never validated on a date earlier than the data it learnt from.
CV_FOLDS = 3
CV_BLOCK_DAYS = 120      # each validation block is ~4 months; the first fold still trains on ~17 months

BLUE, ORANGE, INK, INK_2, INK_3, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def chrono_split(df, share):
    """Split by DATE so one day is never partly in train and partly in test."""
    dates = np.sort(df["date"].unique())
    cut = dates[int(len(dates) * share)]
    return df[df["date"] < cut].copy(), df[df["date"] >= cut].copy(), pd.Timestamp(cut)


def xy(df):
    return df[FEATURES].to_numpy(dtype=float), df[TARGET].to_numpy(dtype=float)


def evaluate(df, pred_col):
    """Three views of accuracy:
       row      - every district-day (inflated: knowing the district already explains most variance)
       network  - total ETS trips per day (the planning-level number KTMB cares about)
       festive  - network days inside a festive period (the surges this project targets)"""
    y, p = df[TARGET], df[pred_col]
    net = df.groupby("date")[[TARGET, pred_col]].sum()
    fest = df[df["is_festive_period"] == 1].groupby("date")[[TARGET, pred_col]].sum()
    return {
        "row_r2": r2(y, p), "row_mae": mae(y, p), "row_rmse": rmse(y, p),
        "network_r2": r2(net[TARGET], net[pred_col]), "network_mae": mae(net[TARGET], net[pred_col]),
        "network_rmse": rmse(net[TARGET], net[pred_col]), "network_mape": mape(net[TARGET], net[pred_col]),
        "festive_network_mape": mape(fest[TARGET], fest[pred_col]) if len(fest) else None,
        "festive_days": int(len(fest)),
    }


def cv_folds(train):
    """Rolling-origin folds over the training dates: (fit_dates_end, val_start, val_end)."""
    dates = np.sort(train["date"].unique())
    folds = []
    for k in range(CV_FOLDS, 0, -1):
        v0 = len(dates) - k * CV_BLOCK_DAYS
        v1 = v0 + CV_BLOCK_DAYS
        folds.append((pd.Timestamp(dates[v0]), pd.Timestamp(dates[v1 - 1])))
    return folds


def tune(train):
    """Grid search with time-series cross-validation, inside the training partition only.
    Number of trees is tuned without refitting: trees are grown one after another from the same
    random stream, so the first 50 trees of a 150-tree forest ARE the 50-tree forest."""
    folds = cv_folds(train)
    print(f"\nTuning with {CV_FOLDS}-fold time-series cross-validation inside TRAIN:")
    fold_rows = []
    for i, (v0, v1) in enumerate(folds, 1):
        n_fit = train["date"].lt(v0).sum()
        print(f"  fold {i}: fit {train['date'].min().date()}..{(v0 - pd.Timedelta(days=1)).date()} "
              f"({n_fit:,} rows) -> validate {v0.date()}..{v1.date()}")
        fold_rows.append({"fold": i, "fit_start": str(train["date"].min().date()),
                          "fit_end": str((v0 - pd.Timedelta(days=1)).date()),
                          "val_start": str(v0.date()), "val_end": str(v1.date()), "fit_rows": int(n_fit)})
    pd.DataFrame(fold_rows).to_csv(OUT / "tuning_folds.csv", index=False)

    combos = list(itertools.product(*TUNE_GRID.values()))
    scores = {}                         # (depth, leaf, trees) -> list of per-fold metric dicts
    t_all = time.time()
    for c, values in enumerate(combos, 1):
        params = dict(zip(TUNE_GRID, values))
        t0 = time.time()
        for v0, v1 in folds:
            fit = train[train["date"] < v0]
            val = train[(train["date"] >= v0) & (train["date"] <= v1)].copy()
            Xf, yf = xy(fit)
            Xv, _ = xy(val)
            rf = RandomForest(n_trees=max(TUNE_TREES), max_features=DEFAULT_PARAMS["max_features"],
                              random_state=42, **params).fit(Xf, yf, FEATURES)
            cum = np.cumsum([t.predict(Xv) for t in rf.trees], axis=0)
            for n in TUNE_TREES:
                val["_p"] = cum[n - 1] / n
                scores.setdefault((*values, n), []).append(evaluate(val, "_p"))
        done = time.time() - t_all
        print(f"  [{c}/{len(combos)}] depth={params['max_depth']}, leaf={params['min_samples_leaf']}: "
              + ", ".join(f"{n} trees MAPE {np.mean([m['network_mape'] for m in scores[(*values, n)]]):.2f}%"
                          for n in TUNE_TREES)
              + f"  ({time.time() - t0:.0f}s, about {done / c * (len(combos) - c) / 60:.0f} min left)")

    rows = []
    for (depth, leaf, n), ms in scores.items():
        row = {"n_trees": n, "max_depth": depth, "min_samples_leaf": leaf}
        for key, col in [("network_mape", "cv_network_mape"), ("network_r2", "cv_network_r2"),
                         ("row_rmse", "cv_row_rmse"), ("row_mae", "cv_row_mae"),
                         ("festive_network_mape", "cv_festive_mape")]:
            vals = [m[key] for m in ms if m[key] is not None]
            row[col] = float(np.mean(vals)) if vals else None
        row["cv_network_mape_sd"] = float(np.std([m["network_mape"] for m in ms]))
        for i, m in enumerate(ms, 1):
            row[f"fold{i}_network_mape"] = m["network_mape"]
        rows.append(row)
    res = pd.DataFrame(rows).sort_values(["cv_network_mape", "n_trees"]).reset_index(drop=True)
    res.to_csv(OUT / "tuning_results.csv", index=False)
    tuning_chart(res)
    best = res.iloc[0]
    chosen = {"n_trees": int(best["n_trees"]), "max_depth": int(best["max_depth"]),
              "min_samples_leaf": int(best["min_samples_leaf"])}
    print(f"\nTop 5 settings (mean over {CV_FOLDS} folds):")
    print(res.head(5)[["n_trees", "max_depth", "min_samples_leaf", "cv_network_mape", "cv_network_r2",
                       "cv_row_rmse", "cv_festive_mape"]].round(3).to_string(index=False))
    print(f"Best setting: {chosen}  (tuning took {(time.time() - t_all) / 60:.0f} min)")
    return chosen


def tuning_chart(res):
    fig, axes = plt.subplots(1, len(TUNE_GRID["min_samples_leaf"]), figsize=(11, 3.8), sharey=True)
    colors = {10: ORANGE, 15: BLUE, 20: INK}
    for ax, leaf in zip(axes, TUNE_GRID["min_samples_leaf"]):
        for depth in TUNE_GRID["max_depth"]:
            g = res[(res["min_samples_leaf"] == leaf) & (res["max_depth"] == depth)].sort_values("n_trees")
            ax.plot(g["n_trees"], g["cv_network_mape"], marker="o", color=colors.get(depth, INK_2),
                    lw=1.6, label=f"max depth {depth}")
        ax.set_title(f"min samples per leaf = {leaf}", fontsize=10, loc="left")
        ax.set_xticks(TUNE_TREES)
        ax.set_xlabel("Number of trees")
    axes[0].set_ylabel("Mean CV network MAPE (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Figure 4.x  Hyperparameter tuning, {CV_FOLDS}-fold time-series cross-validation",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "fig_tuning.png", facecolor="white"); plt.close(fig)


def charts(test, importance, district_err, metrics):
    # feature importance
    imp = pd.Series(importance).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([LABELS[k] for k in imp.index], imp.values * 100, color=BLUE, height=0.65)
    for i, v in enumerate(imp.values * 100):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8, color=INK)
    ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of total impurity reduction (%)")
    ax.set_title("Figure 4.x  Random Forest feature importance (mean decrease in impurity)")
    fig.tight_layout(); fig.savefig(OUT / "fig_importance.png", facecolor="white"); plt.close(fig)

    # network actual vs predicted over the test period
    net = test.groupby("date")[[TARGET, "pred", "naive"]].sum()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    fest = test.groupby("date")["is_festive_period"].max()
    for d in fest[fest == 1].index:
        ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color="#f0efec", zorder=0)
    ax.plot(net.index, net[TARGET], color=INK, lw=1.6, label="Actual")
    ax.plot(net.index, net["pred"], color=BLUE, lw=1.6, label="Random Forest")
    ax.plot(net.index, net["naive"], color=ORANGE, lw=1.1, ls="--", label="Naive (district weekday average)")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
    ax.set_ylabel("ETS trips per day (network)")
    ax.set_title(f"Figure 4.x  Test period: actual vs predicted network ridership "
                 f"(R² {metrics['rf']['network_r2']:.2f}, MAPE {metrics['rf']['network_mape']:.1f}%)")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.text(0.99, 0.02, "shaded = festive period", transform=ax.transAxes, ha="right", fontsize=8, color=INK_3)
    fig.tight_layout(); fig.savefig(OUT / "fig_test_network.png", facecolor="white"); plt.close(fig)

    # district error
    de = district_err.sort_values("pct_error")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([d.replace("W.P. ", "") for d in de["district"]], de["pct_error"], color=BLUE, height=0.65)
    for i, (v, m) in enumerate(zip(de["pct_error"], de["mae"])):
        ax.text(v + 0.4, i, f"{v:.1f}%  (±{m:.0f} trips)", va="center", fontsize=8, color=INK)
    ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, de["pct_error"].max() * 1.35)
    ax.set_xlabel("MAE as % of the district's average daily ridership")
    ax.set_title("Figure 4.x  Test error by district")
    fig.tight_layout(); fig.savefig(OUT / "fig_district_error.png", facecolor="white"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="tune with time-series cross-validation first")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA)
    study = df[df["split"] == "study"]
    train, test, cut = chrono_split(study, TRAIN_SHARE)
    print(f"Train: {train['date'].min().date()} to {(cut - pd.Timedelta(days=1)).date()}  ({len(train):,} rows)")
    print(f"Test : {cut.date()} to {test['date'].max().date()}  ({len(test):,} rows)")
    print(f"Features ({len(FEATURES)}): {', '.join(FEATURES)}")

    params = dict(DEFAULT_PARAMS)
    if args.tune:
        params.update(tune(train))

    print(f"\nTraining Random Forest {params} ...")
    t0 = time.time()
    X_tr, y_tr = xy(train)
    rf = RandomForest(**params).fit(X_tr, y_tr, FEATURES, verbose=True)
    train_secs = time.time() - t0
    print(f"Trained in {train_secs:.0f}s | out-of-bag R2 {rf.oob_r2_:.4f}")

    X_te, _ = xy(test)
    test["pred"] = rf.predict(X_te)
    # naive baseline: each district's average for that weekday in the training period
    naive = train.groupby(["district", "day_of_week"])[TARGET].mean().rename("naive")
    test = test.merge(naive, on=["district", "day_of_week"], how="left")

    metrics = {"rf": evaluate(test, "pred"), "naive_baseline": evaluate(test, "naive")}
    district_err = (test.groupby("district")
                    .apply(lambda g: pd.Series({"mean_ridership": g[TARGET].mean(), "mae": mae(g[TARGET], g["pred"]),
                                                "rmse": rmse(g[TARGET], g["pred"])}), include_groups=False)
                    .reset_index())
    district_err["pct_error"] = district_err["mae"] / district_err["mean_ridership"] * 100

    imp = rf.feature_importance()
    sizes = [t.n_nodes for t in rf.trees]
    summary = {
        "params": params, "tuned": bool(args.tune), "features": FEATURES,
        "train_period": [str(train["date"].min().date()), str((cut - pd.Timedelta(days=1)).date())],
        "test_period": [str(cut.date()), str(test["date"].max().date())],
        "train_rows": len(train), "test_rows": len(test), "train_seconds": round(train_secs, 1),
        "oob_r2": rf.oob_r2_, "mean_nodes_per_tree": float(np.mean(sizes)),
        "metrics": metrics, "feature_importance": imp,
        "median_district_pct_error": float(district_err["pct_error"].median()),
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    test[["date", "state", "district", TARGET, "pred", "naive", "is_festive_period"]].to_csv(
        OUT / "test_predictions.csv", index=False)
    district_err.sort_values("pct_error").to_csv(OUT / "district_errors.csv", index=False)
    rf.save(MODELS / "rf_model.npz")
    # the settings used here are reused by 06 (final model) and 07 (forecast model)
    (MODELS / "best_params.json").write_text(json.dumps({**params, "tuned": bool(args.tune)}, indent=2))
    charts(test, imp, district_err, metrics)

    r, n = metrics["rf"], metrics["naive_baseline"]
    print("\n================ TEST RESULTS ================")
    print(f"{'':28s}{'Random Forest':>15s}{'Naive baseline':>16s}")
    for key, label, fmt in [("row_r2", "Row R2", "{:.4f}"), ("row_mae", "Row MAE (trips)", "{:.1f}"),
                            ("row_rmse", "Row RMSE (trips)", "{:.1f}"),
                            ("network_r2", "Network daily R2", "{:.3f}"),
                            ("network_mae", "Network daily MAE", "{:.0f}"),
                            ("network_mape", "Network daily MAPE %", "{:.2f}"),
                            ("festive_network_mape", "Festive-day MAPE %", "{:.2f}")]:
        print(f"{label:28s}{fmt.format(r[key]):>15s}{fmt.format(n[key]):>16s}")
    gain = (1 - r["festive_network_mape"] / n["festive_network_mape"]) * 100
    print(f"\nFestive-day error reduced by {gain:.0f}% versus the naive baseline.")
    print("Top features: " + ", ".join(f"{LABELS[k]} {v:.1%}" for k, v in
                                       sorted(imp.items(), key=lambda kv: -kv[1])[:5]))
    print(f"Median district error: {summary['median_district_pct_error']:.1f}% of its average ridership")
    print(f"\nSaved model to {MODELS / 'rf_model.npz'} and results to {OUT}")


if __name__ == "__main__":
    main()
