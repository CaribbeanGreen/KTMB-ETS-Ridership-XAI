"""
ets_features.py  -  one place that defines the model inputs, shared by training,
TreeSHAP and the prediction dashboard (so all three always use identical features).
"""
import pandas as pd

TARGET = "ridership"

# Model inputs, in a fixed order. Deliberately EXCLUDED:
#   active_hours, n_destinations   -> computed from the same day's ridership (would leak the answer)
#   days_since_service_start       -> every kept district ran before 2023, so it only counts
#                                     days since the data starts (no real information)
#   year                           -> trees cannot extrapolate a trend into unseen years
#   temp_max_c, rain_hours, is_heavy_rain, hh_income_mean_rm -> near-duplicates of kept features
FEATURES = [
    "day_of_week",          # 0 = Monday ... 6 = Sunday
    "is_weekend",
    "month",
    "is_public_holiday",
    "is_holiday_eve",
    "days_to_festival",
    "days_since_festival",
    "is_festive_period",
    "is_ramadan",
    "temp_mean_c",
    "rain_mm",
    "population_k",
    "hh_income_median_rm",
]

LABELS = {
    "day_of_week": "Day of week", "is_weekend": "Weekend", "month": "Month",
    "is_public_holiday": "Public holiday", "is_holiday_eve": "Holiday eve",
    "days_to_festival": "Days to festival", "days_since_festival": "Days since festival",
    "is_festive_period": "Festive period", "is_ramadan": "Ramadan",
    "temp_mean_c": "Mean temperature", "rain_mm": "Rainfall",
    "population_k": "District population", "hh_income_median_rm": "Median household income",
}


def add_ramadan(df):
    """is_ramadan = 1 from 'Beginning of Ramadan' up to the day before Hari Raya Aidilfitri.
    Dates come from the state holiday calendars already in the dataset."""
    df = df.copy()
    names = df["holiday_name"].fillna("")
    starts = sorted(df.loc[names.str.contains("Ramadan"), "date"].unique())
    raya = sorted(df.loc[names.str.contains("Eid al-Fitr"), "date"].unique())
    df["is_ramadan"] = 0
    for s in starts:
        after = [r for r in raya if r > s]
        if after:
            df.loc[df["date"].between(pd.Timestamp(s), pd.Timestamp(after[0]) - pd.Timedelta(days=1)),
                   "is_ramadan"] = 1
    return df


def load_dataset(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = add_ramadan(df)
    return df.sort_values(["date", "district"]).reset_index(drop=True)
