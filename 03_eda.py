"""
03_eda.py  -  Exploratory Data Analysis (Chapter 4.3)
FYP: An Explainable Prediction Model Using Random Forest for KTMB ETS Ridership Surges

Reads data/processed/ets_district_daily.csv (study years 2023-2025 only; the 2026
holdout is never looked at, so it stays a truly unseen test) and writes:

  data/processed/eda/fig01_daily_trend.png         network ridership over time + festivals + Ramadan
  data/processed/eda/fig02_weekday_month.png       average by day of week and by month
  data/processed/eda/fig03_festive_curve.png       ridership around each festival (days before/after)
  data/processed/eda/fig04_day_type.png            uplift: weekend / holiday eve / holiday / festive / Ramadan
  data/processed/eda/fig05_rain.png                uplift by daily rainfall band
  data/processed/eda/fig06_district_profile.png    ridership vs population and vs household income
  data/processed/eda/fig07_correlation.png         feature correlation heatmap
  data/processed/eda/fig08_distribution.png        target distribution + per-district spread
  data/processed/eda/eda_summary.txt               every number quoted in the findings

Only pandas + matplotlib (description of the data, no modelling).
Run:   python 03_eda.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed" / "ets_district_daily.csv"
OUT = BASE / "data" / "processed" / "eda"

# ---- visual style (validated palette: categorical order, single-hue for one series) ----
BLUE, ORANGE = "#2a78d6", "#eb6834"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID, SHADE = "#e6e5e1", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ef8a7e", "#b8312f"])

FEST_LABEL = {
    "hari_raya_aidilfitri": "Hari Raya Aidilfitri",
    "chinese_new_year": "Chinese New Year",
    "deepavali": "Deepavali",
    "hari_raya_haji": "Hari Raya Haji",
    "christmas": "Christmas",
}
FEST_SHORT = {"hari_raya_aidilfitri": "Raya", "chinese_new_year": "CNY", "deepavali": "Deepavali",
              "hari_raya_haji": "Haji", "christmas": "Xmas"}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white",
})

summary = []


def note(line=""):
    print(line)
    summary.append(line)


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print(f"  saved {name}")


def thousands(x, _=None):
    return f"{x:,.0f}"


# --------------------------------------------------------------------------
def load():
    df = pd.read_csv(DATA, parse_dates=["date"])
    df = df[df["split"] == "study"].copy()
    df["holiday_name"] = df["holiday_name"].fillna("")
    df["festival"] = df["festival"].fillna("")

    # Ramadan = from 'Beginning of Ramadan' to the day before Hari Raya (state calendars carry it)
    starts = sorted(df.loc[df["holiday_name"].str.contains("Ramadan"), "date"].unique())
    raya = sorted(df.loc[df["holiday_name"].str.contains("Eid al-Fitr"), "date"].unique())
    ramadan = []
    for s in starts:
        nxt = [r for r in raya if r > s]
        if nxt:
            ramadan.append((pd.Timestamp(s), pd.Timestamp(nxt[0]) - pd.Timedelta(days=1)))
    df["is_ramadan"] = 0
    for s, e in ramadan:
        df.loc[df["date"].between(s, e), "is_ramadan"] = 1

    # network level: one row per day
    net = df.groupby("date").agg(
        ridership=("ridership", "sum"),
        is_weekend=("is_weekend", "max"),
        is_festive_period=("is_festive_period", "max"),
        is_ramadan=("is_ramadan", "max"),
        days_to_festival=("days_to_festival", "min"),
        days_since_festival=("days_since_festival", "min"),
    ).reset_index()
    net["year"], net["dow"], net["month"] = net["date"].dt.year, net["date"].dt.dayofweek, net["date"].dt.month

    # district baseline: average of that district's normal weekdays (not festive, not holiday, not Ramadan)
    normal = (df["is_weekend"].eq(0) & df["is_public_holiday"].eq(0) & df["is_holiday_eve"].eq(0)
              & df["is_festive_period"].eq(0) & df["is_ramadan"].eq(0))
    base = df[normal].groupby("district")["ridership"].mean().rename("district_base")
    df = df.merge(base, on="district")
    df["uplift_pct"] = (df["ridership"] / df["district_base"] - 1) * 100
    return df, net, ramadan


# --------------------------------------------------------------------------
def fig01_trend(df, net, ramadan):
    fests = (df[df["days_to_festival"].eq(0)][["date", "festival"]].drop_duplicates("date"))
    years = sorted(net["year"].unique())
    fig, axes = plt.subplots(len(years), 1, figsize=(11, 2.6 * len(years)), sharey=True)
    ymax = net["ridership"].max() * 1.12
    ymin = net["ridership"].min() * 0.85
    for ax, y in zip(np.atleast_1d(axes), years):
        n = net[net["year"] == y]
        for s, e in ramadan:
            if s.year == y:
                ax.axvspan(s, e, color=SHADE, zorder=0)
                ax.text(s + (e - s) / 2, ymax * 0.97, "Ramadan", ha="center", va="top", fontsize=8, color=INK_3)
        ax.plot(n["date"], n["ridership"], color=BLUE, lw=1.1)
        ax.plot(n["date"], n["ridership"].rolling(7, center=True).mean(), color=INK, lw=2, alpha=0.75)
        for _, f in fests[fests["date"].dt.year == y].iterrows():
            ax.axvline(f["date"], color=ORANGE, lw=1, ls="--")
            ax.text(f["date"], ymin + (ymax - ymin) * 0.84, " " + FEST_SHORT.get(f["festival"], ""), color=INK_2, fontsize=8)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31"))
        ax.yaxis.set_major_formatter(thousands)
        ax.set_ylabel(str(y), rotation=0, ha="right", va="center", fontsize=11, color=INK, fontweight="bold")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    np.atleast_1d(axes)[0].set_title("Figure 4.x  Daily ETS network ridership, 2023–2025 (trips per day)")
    np.atleast_1d(axes)[-1].plot([], [], color=BLUE, lw=1.1, label="Daily trips")
    np.atleast_1d(axes)[-1].plot([], [], color=INK, lw=2, alpha=0.75, label="7-day average")
    np.atleast_1d(axes)[-1].plot([], [], color=ORANGE, lw=1, ls="--", label="Festival day")
    np.atleast_1d(axes)[-1].legend(loc="lower right", ncol=3, fontsize=8)
    fig.tight_layout()
    save(fig, "fig01_daily_trend.png")

    note("== Fig 1: daily trend ==")
    for y in years:
        n = net[net["year"] == y]["ridership"]
        note(f"{y}: total {n.sum():,} | mean/day {n.mean():,.0f} | min {n.min():,} | max {n.max():,}")
    top = net.nlargest(5, "ridership")
    note("Busiest days: " + "; ".join(f"{d.date()} ({r:,})" for d, r in zip(top["date"], top["ridership"])))
    low = net.nsmallest(5, "ridership")
    note("Quietest days: " + "; ".join(f"{d.date()} ({r:,})" for d, r in zip(low["date"], low["ridership"])))


def fig02_weekday_month(net):
    normal = net[(net["is_festive_period"] == 0) & (net["is_ramadan"] == 0)]
    dow = normal.groupby("dow")["ridership"].mean()
    mon = net.groupby("month")["ridership"].mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [7, 12]})
    for ax, s, labels, title in [(a1, dow, DOW, "By day of week (normal periods)"),
                                 (a2, mon, MONTHS, "By month (all days)")]:
        bars = ax.bar(range(len(s)), s.values, color=BLUE, width=0.7, edgecolor="white", linewidth=1)
        ax.set_xticks(range(len(s)), labels)
        ax.set_grid = True
        ax.yaxis.set_major_formatter(thousands)
        ax.set_title(title, fontsize=10)
        hi, lo = s.values.argmax(), s.values.argmin()
        for i in (hi, lo):
            ax.text(i, s.values[i] + s.max() * 0.015, f"{s.values[i]:,.0f}", ha="center", fontsize=8, color=INK)
        ax.set_ylim(0, s.max() * 1.12)
    a1.set_ylabel("Average trips per day")
    fig.suptitle("Figure 4.x  Average ETS network ridership by day of week and month", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, "fig02_weekday_month.png")
    note("\n== Fig 2: weekly / monthly pattern ==")
    note("Day of week (normal periods): " + ", ".join(f"{DOW[i]} {v:,.0f}" for i, v in dow.items()))
    note(f"Busiest weekday {DOW[dow.idxmax()]} is {(dow.max() / dow.min() - 1) * 100:.0f}% above quietest {DOW[dow.idxmin()]}")
    note("Month: " + ", ".join(f"{MONTHS[i - 1]} {v:,.0f}" for i, v in mon.items()))


def fig03_festive_curve(df, net):
    """Network ridership on each day from 10 before to 10 after a festival, indexed to the
    average for the same weekday in the same year outside festive/Ramadan periods (=100)."""
    normal = net[(net["is_festive_period"] == 0) & (net["is_ramadan"] == 0)]
    ref = normal.groupby(["year", "dow"])["ridership"].mean()
    fests = df[df["days_to_festival"].eq(0)][["date", "festival"]].drop_duplicates("date")
    rows = []
    for _, f in fests.iterrows():
        for k in range(-10, 11):
            d = f["date"] + pd.Timedelta(days=k)
            r = net.loc[net["date"] == d, "ridership"]
            if len(r):
                rows.append({"festival": f["festival"], "offset": k,
                             "index": r.iloc[0] / ref[(d.year, d.dayofweek)] * 100})
    curve = pd.DataFrame(rows).groupby(["festival", "offset"])["index"].mean().reset_index()

    order = ["hari_raya_aidilfitri", "chinese_new_year", "deepavali", "hari_raya_haji", "christmas"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axhline(100, color=INK_3, lw=1)
    ax.axvline(0, color=INK_3, lw=1, ls=":")
    ends = []
    for color, key in zip(CATEGORICAL, order):
        c = curve[curve["festival"] == key]
        if c.empty:
            continue
        ax.plot(c["offset"], c["index"], color=color, lw=2, marker="o", ms=3.5, label=FEST_LABEL[key])
        ends.append([c.iloc[-1]["index"], FEST_LABEL[key]])
    # spread end labels so they never overlap (min gap in index points)
    ends.sort()
    gap = 2.0
    for _ in range(20):                     # push overlapping pairs apart symmetrically
        for i in range(1, len(ends)):
            d = ends[i][0] - ends[i - 1][0]
            if d < gap:
                mid = (ends[i][0] + ends[i - 1][0]) / 2
                ends[i - 1][0], ends[i][0] = mid - gap / 2, mid + gap / 2
    for y, label in ends:
        ax.text(10.3, y, label, color=INK_2, fontsize=8, va="center")
    ax.set_xlim(-10.5, 13.5)
    ax.set_xticks(range(-10, 11, 2))
    ax.set_xlabel("Days relative to festival (0 = first day of the festival)")
    ax.set_ylabel("Ridership index (normal same weekday = 100)")
    ax.set_title("Figure 4.x  ETS ridership before and after each festival, 2023–2025 average")
    ax.legend(loc="upper left", fontsize=8, ncol=5)
    fig.tight_layout()
    save(fig, "fig03_festive_curve.png")

    note("\n== Fig 3: festive curve (index, normal same weekday = 100) ==")
    for key in order:
        c = curve[curve["festival"] == key].set_index("offset")["index"]
        if c.empty:
            continue
        pre, post = c.loc[-10:-1], c.loc[1:10]
        note(f"{FEST_LABEL[key]}: peak before = {pre.max():.0f} (day {pre.idxmax()}), "
             f"festival day = {c.get(0, float('nan')):.0f}, peak after = {post.max():.0f} (day +{post.idxmax()})")


def fig04_day_type(df):
    cats = [
        ("Weekend", df["is_weekend"].eq(1) & df["is_festive_period"].eq(0)),
        ("Holiday eve", df["is_holiday_eve"].eq(1)),
        ("Public holiday", df["is_public_holiday"].eq(1)),
        ("Festive period\n(±7 days)", df["is_festive_period"].eq(1)),
        ("Ramadan", df["is_ramadan"].eq(1) & df["is_festive_period"].eq(0)),
    ]
    vals = [(name, df.loc[m, "uplift_pct"].mean(), int(m.sum())) for name, m in cats]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = [BLUE if v >= 0 else ORANGE for _, v, _ in vals]
    ax.bar(range(len(vals)), [v for _, v, _ in vals], color=colors, width=0.62, edgecolor="white")
    ax.axhline(0, color=INK_3, lw=1)
    for i, (name, v, n) in enumerate(vals):
        ax.text(i, v + (1.2 if v >= 0 else -1.2), f"{v:+.0f}%", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, fontweight="bold", color=INK)
    ax.set_xticks(range(len(vals)), [f"{n}\n(n={c:,})" for n, _, c in vals], fontsize=9)
    ax.set_ylabel("Change vs district's normal weekday")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:+.0f}%")
    lo, hi = min(v for _, v, _ in vals), max(v for _, v, _ in vals)
    ax.set_ylim(min(lo, 0) - 5, hi * 1.25)
    ax.set_title("Figure 4.x  Ridership uplift by day type (district-days, 2023–2025)")
    fig.tight_layout()
    save(fig, "fig04_day_type.png")
    note("\n== Fig 4: uplift vs each district's normal weekday ==")
    for name, v, n in vals:
        note(f"{name.replace(chr(10), ' ')}: {v:+.1f}%  (n={n:,} district-days)")


def fig05_rain(df):
    bands = pd.cut(df["rain_mm"], [-0.01, 0.1, 5, 20, np.inf],
                   labels=["Dry\n(<0.1 mm)", "Light\n(0.1–5 mm)", "Moderate\n(5–20 mm)", "Heavy\n(≥20 mm)"])
    # compare within normal weekdays only, so festive/weekend effects don't leak in
    m = df["is_weekend"].eq(0) & df["is_festive_period"].eq(0) & df["is_public_holiday"].eq(0) & df["is_ramadan"].eq(0)
    g = df[m].groupby(bands[m], observed=True)["uplift_pct"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(range(len(g)), g["mean"], color=BLUE, width=0.6, edgecolor="white")
    ax.axhline(0, color=INK_3, lw=1)
    for i, (v, n) in enumerate(zip(g["mean"], g["count"])):
        ax.text(i, v + (0.3 if v >= 0 else -0.3), "0.0%" if abs(v) < 0.05 else f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9, color=INK)
    ax.set_xticks(range(len(g)), [f"{b}\n(n={int(n):,})" for b, n in zip(g.index, g["count"])], fontsize=9)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:+.0f}%")
    span = max(abs(g["mean"]).max() * 1.6, 3)
    ax.set_ylim(-span, span)
    ax.set_ylabel("Change vs district's normal weekday")
    ax.set_title("Figure 4.x  Ridership by daily rainfall (normal weekdays only)")
    fig.tight_layout()
    save(fig, "fig05_rain.png")
    note("\n== Fig 5: rainfall (normal weekdays) ==")
    for b, (v, n) in g.iterrows():
        note(f"{str(b).replace(chr(10), ' ')}: {v:+.1f}% (n={int(n):,})")


def fig06_district(df):
    d = df.groupby("district").agg(ridership=("ridership", "mean"), population=("population_k", "mean"),
                                   income=("hh_income_median_rm", "mean")).reset_index()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, col, xlabel in [(a1, "population", "District population (thousand)"),
                            (a2, "income", "Median household income (RM / month)")]:
        ax.scatter(d[col], d["ridership"], s=46, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color=GRID)
        r = np.corrcoef(d[col], np.log10(d["ridership"]))[0, 1]
        ax.text(0.02, 0.97, f"r = {r:.2f} (vs log ridership)", transform=ax.transAxes, va="top", fontsize=9, color=INK_2)
        for _, row in d.nlargest(4, "ridership").iterrows():
            left = row["district"] == "Perlis"      # Perlis sits just left of Seberang Perai Utara
            ax.annotate(row["district"].replace("W.P. ", ""), (row[col], row["ridership"]),
                        xytext=(-7 if left else 7, -3), textcoords="offset points",
                        ha="right" if left else "left", fontsize=8, color=INK_2)
        note_r = f"{col}: r = {r:.2f}"
        summary.append(note_r)
    a1.set_ylabel("Average trips per day (log scale)")
    a1.yaxis.set_major_formatter(thousands)
    fig.suptitle("Figure 4.x  District ridership vs population and household income (21 districts)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, "fig06_district_profile.png")
    tot = df.groupby("district")["ridership"].sum().sort_values(ascending=False)
    share = tot / tot.sum() * 100
    note("\n== Fig 6: district profile ==")
    note("Share of study trips: " + ", ".join(f"{k} {v:.1f}%" for k, v in share.head(5).items()))
    note(f"Top 2 districts together: {share.head(2).sum():.1f}%")


def fig07_correlation(df):
    cols = {"ridership": "Ridership (target)", "day_of_week": "Day of week", "is_weekend": "Weekend",
            "month": "Month", "is_public_holiday": "Public holiday", "is_holiday_eve": "Holiday eve",
            "days_to_festival": "Days to festival", "days_since_festival": "Days since festival",
            "is_festive_period": "Festive period", "temp_mean_c": "Mean temp", "rain_mm": "Rainfall",
            "rain_hours": "Rain hours", "population_k": "Population", "hh_income_median_rm": "Median income"}
    # days_since_service_start is left out: every kept district ran before 2023, so it only
    # counts days since the data begins (no real information) - also exclude it from the model
    cols = {k: v for k, v in cols.items() if k in df.columns and df[k].nunique() > 1}
    c = df[list(cols)].corr()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(c.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), [cols[k] for k in c.columns], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(c)), [cols[k] for k in c.index], fontsize=9)
    ax.grid(False)
    for i in range(len(c)):
        for j in range(len(c)):
            v = c.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    ax.set_title("Figure 4.x  Pearson correlation between features (district-days, 2023–2025)")
    fig.tight_layout()
    save(fig, "fig07_correlation.png")
    t = c["ridership"].drop("ridership").sort_values(key=abs, ascending=False)
    note("\n== Fig 7: correlation with ridership ==")
    note(", ".join(f"{cols[k]} {v:+.2f}" for k, v in t.items()))
    pairs = []
    for i, a in enumerate(c.index):
        for b in c.index[i + 1:]:
            if a != "ridership" and b != "ridership" and abs(c.loc[a, b]) >= 0.7:
                pairs.append(f"{cols[a]} ~ {cols[b]} ({c.loc[a, b]:+.2f})")
    note("Highly correlated feature pairs (|r| >= 0.7): " + ("; ".join(pairs) if pairs else "none"))


def fig08_distribution(df, net):
    order = df.groupby("district")["ridership"].median().sort_values().index
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [1, 1.25]})
    a1.hist(net["ridership"], bins=40, color=BLUE, edgecolor="white", linewidth=1)
    a1.axvline(net["ridership"].mean(), color=INK, lw=1.5, ls="--")
    a1.text(net["ridership"].mean(), a1.get_ylim()[1] * 0.97, f"  mean {net['ridership'].mean():,.0f}",
            fontsize=8, color=INK, va="top")
    a1.set_xlabel("Network trips per day")
    a1.set_ylabel("Number of days")
    a1.xaxis.set_major_formatter(thousands)
    a1.set_title("Network daily total", fontsize=10)
    data = [df.loc[df["district"] == d, "ridership"].values for d in order]
    bp = a2.boxplot(data, orientation="horizontal", widths=0.6, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", ms=2, mfc=INK_3, mec="none", alpha=0.5),
                    medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK_3),
                    capprops=dict(color=INK_3))
    for p in bp["boxes"]:
        p.set(facecolor="#b7d3f6", edgecolor=BLUE)
    a2.set_yticks(range(1, len(order) + 1), [d.replace("W.P. ", "") for d in order], fontsize=8)
    a2.set_xscale("log")
    a2.set_xlabel("Trips per day (log scale)")
    a2.xaxis.set_major_formatter(thousands)
    a2.grid(axis="x", color=GRID)
    a2.grid(axis="y", visible=False)
    a2.set_title("Per district (dots = outlier days, mostly festive peaks)", fontsize=10)
    fig.suptitle("Figure 4.x  Distribution of the target variable (ridership), 2023–2025",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, "fig08_distribution.png")
    note("\n== Fig 8: target distribution ==")
    s = df["ridership"]
    note(f"District-day ridership: mean {s.mean():,.0f}, median {s.median():,.0f}, std {s.std():,.0f}, "
         f"skewness {s.skew():.2f}, zero days {int((s == 0).sum())}")
    n = net["ridership"]
    note(f"Network daily: mean {n.mean():,.0f}, std {n.std():,.0f}, skewness {n.skew():.2f}")


def main():
    if not DATA.exists():
        raise SystemExit("Run 02_build_dataset.py first.")
    OUT.mkdir(parents=True, exist_ok=True)
    df, net, ramadan = load()
    note(f"EDA on study period: {df['date'].min().date()} to {df['date'].max().date()}, "
         f"{len(df):,} district-days, {df['district'].nunique()} districts")
    note("Ramadan periods: " + "; ".join(f"{s.date()} to {e.date()}" for s, e in ramadan))
    note("")
    print("Drawing figures...")
    fig01_trend(df, net, ramadan)
    fig02_weekday_month(net)
    fig03_festive_curve(df, net)
    fig04_day_type(df)
    fig05_rain(df)
    fig06_district(df)
    fig07_correlation(df)
    fig08_distribution(df, net)
    (OUT / "eda_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"\nDone. Figures and eda_summary.txt are in {OUT}")


if __name__ == "__main__":
    main()
