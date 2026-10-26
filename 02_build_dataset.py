"""
02_build_dataset.py  -  Phase 2 (Data Collection), step 2 of 2
FYP: An Explainable Prediction Model Using Random Forest for KTMB ETS Ridership Surges

Turns the raw files from 01_collect.py into ONE unified modelling table:

    one row = one departure district x one day
    target  = ridership  (total ETS trips starting in that district that day)

Pipeline
  1. Ridership : hourly O-D records -> map origin station to district -> daily sum
  2. Calendar  : day of week, weekend, state public holidays, festive windows
  3. Weather   : Open-Meteo daily rainfall / temperature for the district
  4. DOSM      : district population + household income (latest year <= row year)
  5. Coverage  : remove pre-opening rows, thin districts, rows without weather
  6. Quality   : gap check, missing values, unmatched names -> data_quality_report.txt

Outputs (data/processed/)
  ets_district_daily.csv   the unified dataset (study years + optional holdout year)
  data_dictionary.csv      every column, its type, source and meaning
  data_quality_report.txt  row counts, gaps, missing values, name-matching results

Run:   python 02_build_dataset.py
"""
import json
import re
from pathlib import Path

import holidays
import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"
LOOKUP = BASE / "station_lookup.csv"

STUDY_YEARS = [2023, 2024, 2025]
HOLDOUT_YEARS = [2026]

# Major festivals that drive ETS "balik kampung" surges (holidays package, en_US names)
FESTIVALS = {
    "Eid al-Fitr": "hari_raya_aidilfitri",
    "Chinese New Year": "chinese_new_year",
    "Diwali": "deepavali",
    "Eid al-Adha": "hari_raya_haji",
    "Christmas": "christmas",
}
FESTIVE_WINDOW = 7          # days either side of a festival counted as "festive period"
DAYS_CAP = 30               # cap for days-to / days-since features
FESTIVAL_MERGE_DAYS = 7     # holiday dates this close together = one festival (e.g. observed days)

# Service coverage rules
MIN_DAILY_TRIPS = 20        # drop districts averaging fewer trips/day than this once in service
EXCLUDE_NEW_LINE = True     # drop the Gemas-JB southern extension (opened in stages during 2025)
NEW_LINE_DISTRICTS = ["Segamat", "Kluang", "Kulai", "Johor Bahru"]
SERVICE_DAYS_CAP = 365      # cap for days_since_service_start

report = []                 # lines for data_quality_report.txt


def log(line=""):
    print(line)
    report.append(line)


def name_key(s):
    """Normalise district spellings so DOSM variants match the lookup.
    e.g. 'S.P.Tengah' == 'Seberang Perai Tengah', 'Larut & Matang' == 'Larut dan Matang',
    'Hulu Langat' == 'Ulu Langat', 'W.P. Kuala Lumpur' == 'Kuala Lumpur'."""
    k = str(s).lower().replace("&", " dan ")
    k = re.sub(r"[^a-z]", "", k)
    k = k.replace("seberangperai", "sp").replace("hulu", "ulu")
    if k.startswith("wp"):
        k = k[2:]
    return k


# --------------------------------------------------------------------------
# 1. Ridership
# --------------------------------------------------------------------------
def build_ridership(lookup):
    log("== 1. RIDERSHIP ==")
    frames = []
    for year in STUDY_YEARS + HOLDOUT_YEARS:
        f = RAW / "ridership" / f"ets_{year}.parquet"
        if not f.exists():
            if year in STUDY_YEARS:
                raise SystemExit(f"Missing {f}. Run 01_collect.py first.")
            continue
        df = pd.read_parquet(f, columns=["date", "time", "origin", "destination", "ridership"])
        df["date"] = pd.to_datetime(df["date"])
        log(f"ets_{year}: {len(df):,} hourly O-D rows, "
            f"{df['date'].min().date()} to {df['date'].max().date()}, "
            f"{df['ridership'].sum():,} trips")
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    # every origin station must map to a district
    stations = set(raw["origin"].unique())
    unmapped = sorted(stations - set(lookup["station"]))
    if unmapped:
        log("\nUNMAPPED STATIONS - add them to station_lookup.csv and re-run:")
        for s in unmapped:
            log(f"   {s!r}  ({int(raw.loc[raw['origin'] == s, 'ridership'].sum()):,} trips)")
        write_report()
        raise SystemExit(1)
    log(f"All {len(stations)} origin stations mapped to "
        f"{lookup['district'].nunique()} districts.")

    # hourly O-D -> daily per origin district
    raw = raw.merge(lookup[["station", "state", "district"]], left_on="origin", right_on="station")

    # Keep only trips WITHIN the established line. Stations whose district first appears after the
    # study starts (the Gemas-JB southern extension, Klang, Rembau) did not exist as destinations
    # in most of the training years; counting trips TO them would make 2026 demand look like a
    # sudden surge in every district (e.g. KL -> JB Sentral trips depart from KL).
    station_district = dict(zip(lookup["station"], lookup["district"]))
    raw["dest_district"] = raw["destination"].map(station_district)
    first_seen = raw[raw["ridership"] > 0].groupby("district")["date"].min()
    new_districts = sorted(first_seen[first_seen > pd.Timestamp(f"{STUDY_YEARS[0]}-01-01")].index)
    to_new = raw["dest_district"].isin(new_districts)
    log(f"\nDistricts that opened during the data period: {new_districts}")
    log("Trips TO those districts removed from the target (not part of the established line):")
    for year, n in raw[to_new].groupby(raw.loc[to_new, "date"].dt.year)["ridership"].sum().items():
        total = raw.loc[raw["date"].dt.year == year, "ridership"].sum()
        log(f"   {year}: {n:,} trips ({n / total:.1%} of that year)")
    raw = raw[~to_new]
    daily = (raw.groupby(["date", "state", "district"], as_index=False)
                .agg(ridership=("ridership", "sum"),
                     active_hours=("time", "nunique"),
                     n_destinations=("destination", "nunique")))

    # full grid so a district with no departures that day shows as 0, not a missing row
    districts = lookup[["state", "district"]].drop_duplicates()
    dates = pd.DataFrame({"date": pd.date_range(daily["date"].min(), daily["date"].max())})
    grid = dates.merge(districts, how="cross")
    daily = grid.merge(daily, on=["date", "state", "district"], how="left")

    # a date where the WHOLE network recorded nothing is a data gap, not zero demand
    network = daily.groupby("date")["ridership"].sum(min_count=1)
    gap_days = network[network.isna() | (network == 0)].index
    if len(gap_days):
        log(f"\nNetwork-wide gap days (no records at all), dropped: {len(gap_days)}")
        for d in gap_days[:20]:
            log(f"   {d.date()}")
        if len(gap_days) > 20:
            log(f"   ... and {len(gap_days) - 20} more")
    daily = daily[~daily["date"].isin(gap_days)].copy()
    for c in ["ridership", "active_hours", "n_destinations"]:
        daily[c] = daily[c].fillna(0).astype(int)
    log(f"District-days after aggregation: {len(daily):,}")
    return daily


# --------------------------------------------------------------------------
# 2. Calendar
# --------------------------------------------------------------------------
def add_calendar(df, lookup):
    log("\n== 2. CALENDAR ==")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek          # 0 = Monday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    years = sorted(df["year"].unique())
    subdiv = lookup.drop_duplicates("state").set_index("state")["holiday_subdiv"].to_dict()

    hol_rows, fest_dates = [], {}
    for state, code in subdiv.items():
        cal = holidays.Malaysia(years=years, subdiv=code, language="en_US")
        for d, name in cal.items():
            hol_rows.append({"date": pd.Timestamp(d), "state": state, "holiday_name": name})
            # festivals taken from the states on the ETS line (Deepavali is not national)
            for key, label in FESTIVALS.items():
                if key in name:
                    fest_dates.setdefault(label, set()).add(pd.Timestamp(d))
    hol = pd.DataFrame(hol_rows).drop_duplicates(["date", "state"])
    hol["date"] = hol["date"].astype("datetime64[ns]")
    df["date"] = df["date"].astype("datetime64[ns]")
    df = df.merge(hol, on=["date", "state"], how="left")
    df["is_public_holiday"] = df["holiday_name"].notna().astype(int)
    df["holiday_name"] = df["holiday_name"].fillna("")

    # eve of a holiday: the travel day before it
    hol_keys = set(zip(hol["date"], hol["state"]))
    df["is_holiday_eve"] = [int((d + pd.Timedelta(days=1), s) in hol_keys) and int((d, s) not in hol_keys)
                            for d, s in zip(df["date"], df["state"])]

    # festive proximity: first day of each festival only
    starts = []
    for label, ds in fest_dates.items():
        ds = sorted(ds)
        for i, d in enumerate(ds):
            if i == 0 or (d - ds[i - 1]).days > FESTIVAL_MERGE_DAYS:
                starts.append((d, label))
    starts = pd.DataFrame(starts, columns=["fest_date", "festival"]).sort_values("fest_date")
    starts["fest_date"] = starts["fest_date"].astype("datetime64[ns]")
    log(f"Festival start dates found: {len(starts)}")
    for _, r in starts.iterrows():
        log(f"   {r['fest_date'].date()}  {r['festival']}")

    uniq = pd.DataFrame({"date": pd.to_datetime(sorted(df["date"].unique())).astype("datetime64[ns]")})
    df["date"] = df["date"].astype("datetime64[ns]")
    nxt = pd.merge_asof(uniq, starts.rename(columns={"fest_date": "next_fest"}),
                        left_on="date", right_on="next_fest", direction="forward")
    prv = pd.merge_asof(uniq, starts.rename(columns={"fest_date": "prev_fest", "festival": "prev_festival"}),
                        left_on="date", right_on="prev_fest", direction="backward")
    uniq["days_to_festival"] = (nxt["next_fest"] - uniq["date"]).dt.days.clip(upper=DAYS_CAP).fillna(DAYS_CAP)
    uniq["days_since_festival"] = (uniq["date"] - prv["prev_fest"]).dt.days.clip(upper=DAYS_CAP).fillna(DAYS_CAP)
    near = uniq["days_to_festival"].le(FESTIVE_WINDOW) | uniq["days_since_festival"].le(FESTIVE_WINDOW)
    uniq["is_festive_period"] = near.astype(int)
    uniq["festival"] = ""
    uniq.loc[uniq["days_to_festival"].le(FESTIVE_WINDOW), "festival"] = nxt["festival"]
    since_mask = uniq["days_since_festival"].le(FESTIVE_WINDOW) & uniq["festival"].eq("")
    uniq.loc[since_mask, "festival"] = prv.loc[since_mask, "prev_festival"]
    uniq[["days_to_festival", "days_since_festival"]] = uniq[["days_to_festival", "days_since_festival"]].astype(int)
    df = df.merge(uniq, on="date", how="left")
    log(f"Public-holiday district-days: {df['is_public_holiday'].sum():,}  |  "
        f"festive-period district-days: {df['is_festive_period'].sum():,}")
    return df


# --------------------------------------------------------------------------
# 3. Weather
# --------------------------------------------------------------------------
def add_weather(df):
    log("\n== 3. WEATHER ==")
    frames = []
    for f in sorted((RAW / "weather").glob("weather_*.json")):
        p = json.loads(f.read_text())
        w = pd.DataFrame(p["daily"]).rename(columns={
            "time": "date",
            "temperature_2m_mean": "temp_mean_c",
            "temperature_2m_max": "temp_max_c",
            "precipitation_sum": "rain_mm",
            "precipitation_hours": "rain_hours",
        })
        w["date"] = pd.to_datetime(w["date"])
        w["district"] = p["district"]
        frames.append(w)
    if not frames:
        raise SystemExit("No weather files. Run 01_collect.py first.")
    weather = pd.concat(frames, ignore_index=True)
    missing = sorted(set(df["district"]) - set(weather["district"]))
    if missing:
        log(f"Districts with NO weather file (re-run 01_collect.py): {missing}")
    df = df.merge(weather, on=["date", "district"], how="left")
    df["is_heavy_rain"] = (df["rain_mm"] >= 20).astype(int)   # MetMalaysia 'heavy' threshold, daily
    log(f"Weather attached for {df['rain_mm'].notna().mean():.1%} of district-days")
    return df


# --------------------------------------------------------------------------
# 4. DOSM socio-economics
# --------------------------------------------------------------------------
def latest_by_year(table, value_cols, years):
    """For each district and each study year, take the latest DOSM value from a year <= it."""
    table = table.copy()
    table["src_year"] = pd.to_datetime(table["date"]).dt.year
    rows = []
    for key, g in table.groupby("key"):
        g = g.sort_values("src_year")
        for y in years:
            past = g[g["src_year"] <= y]
            pick = past.iloc[-1] if len(past) else g.iloc[0]
            rows.append({"key": key, "year": y, "src_year": int(pick["src_year"]),
                         **{c: pick[c] for c in value_cols}})
    return pd.DataFrame(rows)


def add_dosm(df):
    log("\n== 4. DOSM ==")
    years = sorted(df["year"].unique())
    df["key"] = df["district"].map(name_key)

    pop = pd.read_parquet(RAW / "dosm" / "population_district.parquet")
    for col, val in [("sex", "both"), ("age", "overall"), ("ethnicity", "overall")]:
        if col in pop.columns:
            pop = pop[pop[col] == val]
    pop["key"] = pop["district"].map(name_key)
    pop = latest_by_year(pop, ["population"], years).rename(
        columns={"population": "population_k", "src_year": "population_src_year"})

    inc = pd.read_parquet(RAW / "dosm" / "hh_income_district.parquet")
    inc["key"] = inc["district"].map(name_key)
    inc = latest_by_year(inc, ["income_mean", "income_median"], years).rename(
        columns={"income_mean": "hh_income_mean_rm", "income_median": "hh_income_median_rm",
                 "src_year": "income_src_year"})

    for name, t in [("population", pop), ("income", inc)]:
        miss = sorted(set(df.loc[~df["key"].isin(t["key"]), "district"]))
        log(f"{name}: matched {df['key'].isin(t['key']).mean():.0%} of rows"
            + (f"; UNMATCHED districts: {miss}" if miss else ""))
    df = df.merge(pop, on=["key", "year"], how="left").merge(inc, on=["key", "year"], how="left")
    log("Source year used per study year (population / income):")
    for y in years:
        s = df[df["year"] == y]
        pop_y = sorted(int(v) for v in s["population_src_year"].dropna().unique())
        inc_y = sorted(int(v) for v in s["income_src_year"].dropna().unique())
        log(f"   {y}: population {pop_y}  income {inc_y}")
    return df.drop(columns="key")


# --------------------------------------------------------------------------
# 5. Service coverage + completeness
# --------------------------------------------------------------------------
def apply_service_rules(df):
    """A district with no train service cannot have demand. Rows before a district's
    first recorded departure are structural zeros (station not open yet), not low
    demand, so they are removed instead of being taught to the model as 'zero'."""
    log("\n== 5. SERVICE COVERAGE ==")
    start = df[df["ridership"] > 0].groupby("district")["date"].min()
    df = df.merge(start.rename("service_start"), on="district", how="left")
    before = df["service_start"].isna() | (df["date"] < df["service_start"])
    log("First recorded departure per district (only those after 2023-01-01 shown):")
    for dist, d in start.sort_values().items():
        if d > pd.Timestamp("2023-01-01"):
            log(f"   {dist}: {d.date()}")
    log(f"Pre-service district-days removed (structural zeros): {before.sum():,}")
    df = df[~before].copy()
    df["days_since_service_start"] = (df["date"] - df["service_start"]).dt.days.clip(upper=SERVICE_DAYS_CAP)

    study = df[df["split"] == "study"]
    no_study = sorted(set(df["district"]) - set(study["district"]))
    avg = study.groupby("district")["ridership"].mean()
    thin = sorted(avg[avg < MIN_DAILY_TRIPS].index)
    drop = set(no_study) | set(thin)
    if no_study:
        log(f"Dropped - no service in study years: {no_study}")
    if thin:
        log(f"Dropped - under {MIN_DAILY_TRIPS} trips/day on average: "
            + ", ".join(f"{d} ({avg[d]:.1f})" for d in thin))
    if EXCLUDE_NEW_LINE:
        new = [d for d in NEW_LINE_DISTRICTS if d in set(df["district"])]
        log(f"Dropped - southern extension, under 1 year of study history: {new}")
        drop |= set(new)
    df = df[~df["district"].isin(drop)].copy()

    no_weather = df["rain_mm"].isna()
    if no_weather.any():
        log(f"Dropped - weather not yet published (archive lags ~1 week): {no_weather.sum()} rows, "
            f"{df.loc[no_weather, 'date'].min().date()} to {df.loc[no_weather, 'date'].max().date()}")
        df = df[~no_weather].copy()
    log(f"Districts kept: {df['district'].nunique()}")
    return df.drop(columns="service_start")


# --------------------------------------------------------------------------
# 6. Output
# --------------------------------------------------------------------------
DICTIONARY = [
    ("date", "date", "derived", "Calendar day (Asia/Kuala_Lumpur)"),
    ("state", "text", "lookup", "State of the departure district"),
    ("district", "text", "lookup", "DOSM administrative district of the departure station"),
    ("split", "text", "derived", "'study' (2023-2025, train/test) or 'holdout' (2026, unseen check)"),
    ("ridership", "int", "data.gov.my ridership_od_ets", "TARGET: ETS trips departing this district this day to stations on the established line (Padang Besar-Gemas)"),
    ("active_hours", "int", "data.gov.my ridership_od_ets", "Hours of the day with at least one departure"),
    ("n_destinations", "int", "data.gov.my ridership_od_ets", "Distinct destination stations served that day"),
    ("days_since_service_start", "int", "derived", f"Days since the district's first ETS departure, capped at {SERVICE_DAYS_CAP}"),
    ("year", "int", "derived", "Calendar year"),
    ("month", "int", "derived", "Month 1-12"),
    ("day_of_week", "int", "derived", "0 = Monday ... 6 = Sunday"),
    ("is_weekend", "0/1", "derived", "Saturday or Sunday"),
    ("holiday_name", "text", "holidays package", "Name of the public holiday in this state, blank if none"),
    ("is_public_holiday", "0/1", "holidays package", "Federal or state public holiday in this state"),
    ("is_holiday_eve", "0/1", "derived", "Day before a public holiday (typical outbound travel day)"),
    ("days_to_festival", "int", "derived", f"Days until the next major festival, capped at {DAYS_CAP}"),
    ("days_since_festival", "int", "derived", f"Days since the last major festival, capped at {DAYS_CAP}"),
    ("is_festive_period", "0/1", "derived", f"Within {FESTIVE_WINDOW} days of a major festival"),
    ("festival", "text", "derived", "Which festival the festive period belongs to"),
    ("temp_mean_c", "float", "Open-Meteo archive", "Daily mean air temperature at 2 m (deg C)"),
    ("temp_max_c", "float", "Open-Meteo archive", "Daily max air temperature at 2 m (deg C)"),
    ("rain_mm", "float", "Open-Meteo archive", "Daily total precipitation (mm)"),
    ("rain_hours", "float", "Open-Meteo archive", "Hours with precipitation"),
    ("is_heavy_rain", "0/1", "derived", "rain_mm >= 20"),
    ("population_k", "float", "DOSM population_district", "District population, thousands (latest year <= row year)"),
    ("population_src_year", "int", "DOSM population_district", "DOSM year the population value comes from"),
    ("hh_income_mean_rm", "float", "DOSM hh_income_district", "Mean gross monthly household income (RM)"),
    ("hh_income_median_rm", "float", "DOSM hh_income_district", "Median gross monthly household income (RM)"),
    ("income_src_year", "int", "DOSM hh_income_district", "HIES year the income value comes from"),
]


def write_report():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data_quality_report.txt").write_text("\n".join(report), encoding="utf-8")


def main():
    lookup = pd.read_csv(LOOKUP, dtype={"holiday_subdiv": str})
    df = build_ridership(lookup)
    df = add_calendar(df, lookup)
    df = add_weather(df)
    df = add_dosm(df)
    df["split"] = df["year"].map(lambda y: "study" if y in STUDY_YEARS else "holdout")
    df = apply_service_rules(df)

    cols = [c for c, *_ in DICTIONARY]
    df = df[cols].sort_values(["date", "state", "district"]).reset_index(drop=True)

    log("\n== 6. FINAL DATASET ==")
    for split, g in df.groupby("split"):
        log(f"{split}: {len(g):,} rows | {g['date'].min().date()} to {g['date'].max().date()} "
            f"| {g['district'].nunique()} districts | {g['ridership'].sum():,} trips")
    log("\nMissing values per column (only columns with gaps):")
    na = df.isna().sum()
    for c, n in na[na > 0].items():
        log(f"   {c}: {n:,} ({n / len(df):.1%})")
    if not (na > 0).any():
        log("   none")
    log("\nRidership by district (study years):")
    s = df[df["split"] == "study"].groupby("district")["ridership"].agg(["sum", "mean", "max"])
    log(s.sort_values("sum", ascending=False).round(1).to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ets_district_daily.csv", index=False)
    pd.DataFrame(DICTIONARY, columns=["column", "type", "source", "description"]).to_csv(
        OUT / "data_dictionary.csv", index=False)
    write_report()
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
