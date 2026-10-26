"""demo_explain.py - live demo: predict and explain one district-day.
Usage: python demo_explain.py "W.P. Kuala Lumpur" 2025-03-28 [trees]"""
import sys
import numpy as np
from ets_features import FEATURES, LABELS, TARGET, load_dataset
from ets_rf import RandomForest
from ets_treeshap import forest_shap

district, date = sys.argv[1], sys.argv[2]
n_trees = int(sys.argv[3]) if len(sys.argv) > 3 else None

rf = RandomForest.load("models/rf_model.npz")
if n_trees:                                   # try a smaller forest live
    rf.trees = rf.trees[:n_trees]
df = load_dataset("data/processed/ets_district_daily.csv")
row = df[(df["district"] == district) & (df["date"] == date)]
x = row[FEATURES].to_numpy(float)

pred = rf.predict(x)[0]
phi, base = forest_shap(rf, x)
print(f"{district}, {date}  ({len(rf.trees)} trees)")
print(f"actual {row[TARGET].iloc[0]:,}   predicted {pred:,.1f}   base value {base:,.1f}")
for j in np.argsort(-np.abs(phi[0])):
    print(f"  {LABELS[FEATURES[j]]:26s} = {x[0, j]:>9g}   {phi[0, j]:+9.1f}")
print(f"check: base + sum(SHAP) = {base + phi.sum():,.4f}  vs prediction {pred:,.4f}")
