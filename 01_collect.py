"""
01_collect.py  -  Phase 2 (Data Collection), step 1 of 2
FYP: An Explainable Prediction Model Using Random Forest for KTMB ETS Ridership Surges

Downloads every raw source into data/raw/ and never modifies them.
Re-running skips files that are already downloaded (delete a file to re-fetch it).

Sources (all open data, CC BY 4.0 unless stated):
  1. KTMB ETS hourly origin-destination ridership   data.gov.my  (ridership_od_ets)
  2. District population, 2020-2024                  DOSM         (population_district)
  3. District household income (HIES), 2019-2024     DOSM         (hh_income_district)
  4. Daily historical weather per district           Open-Meteo archive API (CC BY 4.0)
  5. Public holidays per state                       python `holidays` package (data only)

Run:   python 01_collect.py            (first time: downloads everything)
       python 01_collect.py --latest   (later: re-downloads this year's ridership and the weather,
                                        to pick up days published since the last run)
Needs: pip install pandas pyarrow requests holidays
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
YEARS = [2023, 2024, 2025]          # study period (2022 excluded: post-pandemic distortion)
HOLDOUT_YEARS = [2026]              # optional unseen year for a final real-world check

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
LOOKUP = BASE / "station_lookup.csv"

ETS_URL = "https://storage.data.gov.my/transportation/ktmb/ets_{year}.parquet"
POP_URL = "https://storage.dosm.gov.my/population/population_district.parquet"
INCOME_URL = "https://storage.dosm.gov.my/hies/hh_income_district.parquet"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "precipitation_sum",
    "precipitation_hours",
]


def download(url, dest, label, force=False):
    """Stream a file to disk. Returns True if the file is present afterwards.
    force=True re-downloads an existing file; the old copy is kept if the download fails."""
    if dest.exists() and not force:
        print(f"  [skip] {label}: already downloaded")
        return True
    print(f"  [get ] {label}")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            if r.status_code == 404:
                print(f"         not published (404): {url}")
                return False
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
    except requests.RequestException as e:
        print(f"         FAILED: {e}")
        return False
    print(f"         saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


LATEST = False                      # set by --latest: refresh files that grow over time


def collect_ridership():
    print("\n[1/4] KTMB ETS ridership (hourly origin-destination)")
    out = RAW / "ridership"
    out.mkdir(parents=True, exist_ok=True)
    got = []
    for year in YEARS + HOLDOUT_YEARS:
        refresh = LATEST and year == pd.Timestamp.today().year
        if download(ETS_URL.format(year=year), out / f"ets_{year}.parquet", f"ets_{year}", force=refresh):
            got.append(year)
    missing = [y for y in YEARS if y not in got]
    if missing:
        raise SystemExit(f"Study years missing: {missing}. Check the URL on data.gov.my.")
    return got


def collect_dosm():
    print("\n[2/4] DOSM population by district")
    out = RAW / "dosm"
    out.mkdir(parents=True, exist_ok=True)
    download(POP_URL, out / "population_district.parquet", "population_district")
    print("\n[3/4] DOSM household income by district")
    download(INCOME_URL, out / "hh_income_district.parquet", "hh_income_district")


def collect_weather(years):
    """One daily weather series per district, taken at the coordinates of the
    district's busiest ETS station (the first station listed for it)."""
    print("\n[4/4] Open-Meteo daily weather per district")
    out = RAW / "weather"
    out.mkdir(parents=True, exist_ok=True)
    lookup = pd.read_csv(LOOKUP, dtype={"holiday_subdiv": str})
    points = lookup.drop_duplicates("district")[["district", "lat", "lon"]]

    start = f"{min(years)}-01-01"
    # the archive only covers the past; stop a week ago to avoid half-filled days
    end = min(pd.Timestamp(f"{max(years)}-12-31"), pd.Timestamp.today().normalize() - pd.Timedelta(days=7))
    end = end.strftime("%Y-%m-%d")

    for _, p in points.iterrows():
        slug = p["district"].lower().replace(".", "").replace(" ", "_")
        dest = out / f"weather_{slug}.json"
        if dest.exists() and not LATEST:
            print(f"  [skip] {p['district']}")
            continue
        params = {
            "latitude": p["lat"], "longitude": p["lon"],
            "start_date": start, "end_date": end,
            "daily": ",".join(WEATHER_VARS),
            "timezone": "Asia/Kuala_Lumpur",
        }
        for attempt in range(3):
            try:
                r = requests.get(WEATHER_URL, params=params, timeout=60)
                if r.status_code == 429:           # rate limited: back off
                    time.sleep(30 * (attempt + 1))
                    continue
                r.raise_for_status()
                payload = r.json()
                payload["district"] = p["district"]
                tmp = dest.with_suffix(".part")
                tmp.write_text(json.dumps(payload))
                tmp.replace(dest)
                print(f"  [get ] {p['district']}: {len(payload['daily']['time'])} days")
                break
            except requests.RequestException as e:
                print(f"         retry {attempt + 1}/3 for {p['district']}: {e}")
                time.sleep(5)
        else:
            print(f"         FAILED: {p['district']} (re-run the script later)")
        time.sleep(1)                               # be polite to the free API


if __name__ == "__main__":
    import sys
    LATEST = "--latest" in sys.argv
    if LATEST:
        print("--latest: re-downloading this year's ridership file and all weather files")
    RAW.mkdir(parents=True, exist_ok=True)
    years = collect_ridership()
    collect_dosm()
    collect_weather(years)
    print("\nDone. Raw files are in", RAW)
    print("Next: python 02_build_dataset.py")
