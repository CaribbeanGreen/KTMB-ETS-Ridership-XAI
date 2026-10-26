"""
05_explain.py  -  TreeSHAP explanation module (Chapter 4.7, explainability validation)
FYP: An Explainable Prediction Model Using Random Forest for KTMB ETS Ridership Surges

Uses the trained forest (models/rf_model.npz from 04_train_model.py) and ets_treeshap.py.
Produces the three outputs promised in Table 1.1:

  1. GLOBAL  - summary plot: which features drive predictions overall
               fig_shap_importance.png  (mean |SHAP|, all features)
               fig_shap_summary.png     (every explained row as a dot, calendar & weather features)
  2. LOCAL   - force/waterfall plot for a known surge event (Hari Raya exodus, KL, 28 Mar 2025)
               plus the busiest festive day in the test period
               fig_shap_local_raya.png, fig_shap_local_test.png
  3. ADDITIVITY - base value + sum of SHAP values == model prediction, for every explained row

Plus correctness checks (the sanity check also explains ~120 extra study-period rows:
Ramadan days, holiday eves and pre-festival days, so every check has enough cases):
  - TreeSHAP agrees with exact Shapley enumeration (all 8,192 feature subsets) on 3 rows
  - sanity: holiday-eve / festive features push predictions UP on those days, Ramadan pushes DOWN

Run:   python 05_explain.py              (explains 300 test rows, about 2-4 minutes)
       python 05_explain.py --rows 600   (more rows = smoother summary plot, slower)
Outputs go to data/processed/shap/
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from ets_features import FEATURES, LABELS, TARGET, load_dataset
from ets_rf import RandomForest
from ets_treeshap import exact_shapley, forest_shap

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed" / "ets_district_daily.csv"
MODEL = BASE / "models" / "rf_model.npz"
OUT = BASE / "data" / "processed" / "shap"

KNOWN_SURGE = ("2025-03-28", "W.P. Kuala Lumpur")      # Hari Raya Aidilfitri exodus (3 days before Raya)
DISTRICT_FEATURES = ["population_k", "hh_income_median_rm"]   # constant within a district
TEMPORAL = [f for f in FEATURES if f not in DISTRICT_FEATURES]

BLUE, ORANGE, INK, INK_2, INK_3, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
UP, DOWN = "#e34948", "#2a78d6"                          # pushes prediction up / down
LOWHIGH = LinearSegmentedColormap.from_list("lowhigh", ["#2a78d6", "#c3c2b7", "#e34948"])
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "x",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

report = []


def note(line=""):
    print(line)
    report.append(line)


def describe(feature, value):
    """Human-readable feature value for chart labels."""
    if feature == "day_of_week":
        return DOW[int(value)]
    if feature in ("is_weekend", "is_public_holiday", "is_holiday_eve", "is_festive_period", "is_ramadan"):
        return "yes" if value else "no"
    if feature == "month":
        return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][int(value) - 1]
    if feature in ("days_to_festival", "days_since_festival"):
        return f"{int(value)} days" + (" (capped)" if value >= 30 else "")
    if feature == "temp_mean_c":
        return f"{value:.1f} °C"
    if feature == "rain_mm":
        return f"{value:.1f} mm"
    if feature == "population_k":
        return f"{value:,.0f}k"
    if feature == "hh_income_median_rm":
        return f"RM {value:,.0f}"
    return f"{value:g}"


# --------------------------------------------------------------------------
def local_plot(row, phi, base, pred, title, fname, top=8):
    """Waterfall: start at the base value, add each feature's SHAP value, end at the prediction."""
    order = np.argsort(-np.abs(phi))
    keep, rest = order[:top], order[top:]
    items = [(f"{LABELS[FEATURES[i]]} = {describe(FEATURES[i], row[FEATURES[i]])}", phi[i]) for i in keep]
    if len(rest):
        items.append((f"{len(rest)} other features", phi[rest].sum()))
    items = items[::-1]                                    # biggest at the top

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(items) + 1.8))
    running = base + sum(v for _, v in items)              # = prediction; walk back from the top
    pos = pred
    starts = []
    for _, v in items[::-1]:
        starts.append(pos - v)
        pos -= v
    starts = starts[::-1]
    span = max(abs(v) for _, v in items)
    for y, ((label, v), s) in enumerate(zip(items, starts)):
        ax.barh(y, v, left=s, color=UP if v >= 0 else DOWN, height=0.6)
        ax.text(s + v + (span * 0.02 if v >= 0 else -span * 0.02), y, f"{v:+,.0f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.set_yticks(range(len(items)), [lab for lab, _ in items], fontsize=9)
    ax.axvline(base, color=INK_3, lw=1, ls=":")
    ax.axvline(pred, color=INK, lw=1.2)
    lo = min(min(starts), min(s + v for (_, v), s in zip(items, starts)), base, pred)
    hi = max(max(starts), max(s + v for (_, v), s in zip(items, starts)), base, pred)
    pad = (hi - lo) * 0.18
    ax.set_xlim(lo - pad, hi + pad)
    ax.text(base, len(items) - 0.35, f" base value\n {base:,.0f}", fontsize=8, color=INK_3, va="bottom")
    ax.text(pred, len(items) - 0.35, f"prediction {pred:,.0f} \nactual {row[TARGET]:,.0f} ", fontsize=9,
            color=INK, ha="right", va="bottom", fontweight="bold")
    ax.set_ylim(-0.7, len(items) + 0.6)
    ax.set_xlabel("ETS trips departing the district (that day)")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
    ax.set_title(title)
    ax.plot([], [], color=UP, lw=6, label="pushes prediction up")
    ax.plot([], [], color=DOWN, lw=6, label="pushes prediction down")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / fname, facecolor="white")
    plt.close(fig)


def global_plots(rows, phi):
    mean_abs = pd.Series(np.abs(phi).mean(axis=0), index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = [INK_3 if f in DISTRICT_FEATURES else BLUE for f in mean_abs.index]
    ax.barh([LABELS[f] for f in mean_abs.index], mean_abs.values, color=colors, height=0.65)
    for i, v in enumerate(mean_abs.values):
        ax.text(v * 1.12, i, f"{v:,.1f}", va="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.8, mean_abs.max() * 3)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
    ax.set_xlabel("Mean |SHAP value| (trips, log scale)")
    ax.set_title("Figure 4.x  Global feature importance from TreeSHAP")
    ax.plot([], [], color=INK_3, lw=6, label="district profile (fixed per district)")
    ax.plot([], [], color=BLUE, lw=6, label="calendar & weather (changes day to day)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_shap_importance.png", facecolor="white"); plt.close(fig)

    # beeswarm-style summary for the day-to-day features
    order = mean_abs[TEMPORAL].sort_values().index
    fig, ax = plt.subplots(figsize=(10, 6.5))
    rng = np.random.default_rng(0)
    for y, f in enumerate(order):
        j = FEATURES.index(f)
        vals = rows[f].to_numpy(dtype=float)
        rank = (vals - vals.min()) / (vals.max() - vals.min()) if vals.max() > vals.min() else np.zeros_like(vals)
        ax.scatter(phi[:, j], y + rng.uniform(-0.28, 0.28, len(vals)), c=rank, cmap=LOWHIGH, s=9, alpha=0.8,
                   linewidths=0)
    ax.axvline(0, color=INK_3, lw=1)
    ax.set_yticks(range(len(order)), [LABELS[f] for f in order])
    ax.set_xlabel("SHAP value: change in predicted trips caused by the feature")
    ax.set_title("Figure 4.x  TreeSHAP summary plot (calendar & weather features)")
    sm = plt.cm.ScalarMappable(cmap=LOWHIGH)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"]); cb.set_label("feature value", color=INK_2); cb.outline.set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_shap_summary.png", facecolor="white"); plt.close(fig)
    return mean_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=300, help="test rows to explain for the global plots")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rf = RandomForest.load(MODEL)
    df = load_dataset(DATA)
    study = df[df["split"] == "study"]
    dates = np.sort(study["date"].unique())
    cut = dates[int(len(dates) * 0.8)]                    # same chronological split as 04_train_model.py
    test = study[study["date"] >= cut]

    # rows to explain: a random sample of test rows + the two local cases
    sample = test.sample(n=min(args.rows, len(test)), random_state=42)
    raya = df[(df["date"] == KNOWN_SURGE[0]) & (df["district"] == KNOWN_SURGE[1])]
    fest_test = test[test["is_festive_period"] == 1]
    busiest_day = fest_test.groupby("date")[TARGET].sum().idxmax()
    test_case = fest_test[(fest_test["date"] == busiest_day)].nlargest(1, TARGET)
    # extra rows from the whole study period so each sanity check has enough cases
    # (the test period Jun-Dec 2025 contains no Ramadan and few holiday eves)
    sanity_pool = [study[study["is_ramadan"] == 1], study[study["is_holiday_eve"] == 1],
                   study[study["days_to_festival"].between(1, 7)]]
    extra = pd.concat([p.sample(n=min(40, len(p)), random_state=7) for p in sanity_pool])
    rows = pd.concat([raya, test_case, sample, extra]).reset_index(drop=True)
    g0, g1 = 2, 2 + len(sample)                            # positions of the random test sample

    note(f"Explaining {len(rows)} rows with TreeSHAP ({len(rf.trees)} trees)...")
    t0 = time.time()
    X = rows[FEATURES].to_numpy(dtype=float)
    phi, base = forest_shap(rf, X, verbose=True)
    secs = time.time() - t0
    pred = rf.predict(X)
    note(f"Done in {secs:.0f}s. Base value (average training prediction) = {base:,.1f} trips")

    # ---- check 1: additivity
    err = np.abs(base + phi.sum(axis=1) - pred)
    note("\n== CHECK 1: additivity (base + sum of SHAP = prediction) ==")
    note(f"max error over {len(rows)} rows: {err.max():.2e} trips  ->  {'PASS' if err.max() < 1e-6 else 'FAIL'}")

    # ---- check 2: TreeSHAP vs exact enumeration
    note("\n== CHECK 2: TreeSHAP vs exact Shapley enumeration (2^13 = 8,192 subsets) ==")
    worst = 0.0
    for i in range(3):
        t1 = time.time()
        ex, ex_base = exact_shapley(rf, X[i])
        d = np.abs(ex - phi[i]).max()
        worst = max(worst, d)
        note(f"row {i} ({rows.loc[i, 'district']}, {rows.loc[i, 'date'].date()}): max difference {d:.2e} "
             f"({time.time() - t1:.0f}s)")
    note(f"-> {'PASS' if worst < 1e-6 else 'FAIL'} (differences are floating-point rounding only)")

    # ---- check 3: sanity of directions
    note("\n== CHECK 3: do the explanations make sense? ==")
    S = pd.DataFrame(phi, columns=FEATURES)
    checks = [
        ("Holiday eve pushes UP on holiday-eve days", rows["is_holiday_eve"] == 1, "is_holiday_eve", 1),
        ("Days-to-festival pushes UP in the week before a festival",
         rows["days_to_festival"].between(1, 7), "days_to_festival", 1),
        ("Weekend pushes UP on weekends", rows["is_weekend"] == 1, "is_weekend", 1),
        ("Day of week pushes DOWN on Tue/Wed", rows["day_of_week"].isin([1, 2]), "day_of_week", -1),
        ("Ramadan pushes DOWN during Ramadan", rows["is_ramadan"] == 1, "is_ramadan", -1),
    ]
    sanity = []
    for text, mask, f, sign in checks:
        if mask.sum() == 0:
            note(f"{text}: no rows in the explained set (skipped)")
            continue
        m = S.loc[mask.to_numpy(), f].mean()
        ok = np.sign(m) == sign
        sanity.append({"check": text, "rows": int(mask.sum()), "mean_shap": float(m), "pass": bool(ok)})
        note(f"{text}: mean SHAP {m:+.1f} trips over {int(mask.sum())} rows -> {'PASS' if ok else 'FAIL'}")

    # ---- plots
    mean_abs = global_plots(rows.iloc[g0:g1], phi[g0:g1])  # global = the random test sample only
    note("\n== GLOBAL: mean |SHAP| (trips) ==")
    for f, v in mean_abs.sort_values(ascending=False).items():
        note(f"  {LABELS[f]:28s} {v:8.1f}")
    for i, (fname, title) in enumerate([
            ("fig_shap_local_raya.png", f"Figure 4.x  Why the model predicts a surge: Kuala Lumpur, "
                                        f"{pd.Timestamp(KNOWN_SURGE[0]):%d %b %Y} (Hari Raya exodus)"),
            ("fig_shap_local_test.png", f"Figure 4.x  {rows.loc[1, 'district'].replace('W.P. ', '')}, "
                                        f"{rows.loc[1, 'date']:%d %b %Y} (busiest festive day in test period)")]):
        local_plot(rows.loc[i], phi[i], base, pred[i], title, fname)
        top = np.argsort(-np.abs(phi[i]))[:5]
        note(f"\nLOCAL {fname}: predicted {pred[i]:,.0f}, actual {rows.loc[i, TARGET]:,.0f}")
        for j in top:
            note(f"  {LABELS[FEATURES[j]]:28s} = {describe(FEATURES[j], X[i, j]):>14s}   {phi[i, j]:+8.1f}")

    out = rows[["date", "district", TARGET]].copy()
    out["prediction"], out["base_value"] = pred, base
    for j, f in enumerate(FEATURES):
        out[f"shap_{f}"] = phi[:, j]
    out.to_csv(OUT / "shap_values.csv", index=False)
    (OUT / "shap_report.txt").write_text("\n".join(report), encoding="utf-8")
    (OUT / "shap_checks.json").write_text(json.dumps({
        "rows_explained": len(rows), "seconds": round(secs, 1), "base_value": base,
        "additivity_max_error": float(err.max()), "exact_vs_treeshap_max_diff": float(worst),
        "sanity_checks": sanity, "mean_abs_shap": mean_abs.to_dict()}, indent=2))
    print(f"\nSaved figures, shap_values.csv and shap_report.txt to {OUT}")


if __name__ == "__main__":
    main()
