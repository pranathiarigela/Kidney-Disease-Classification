import numpy as np, json, os
from pathlib import Path

p = Path("data/processed")
files = ["X_train.npy","y_train.npy","X_val.npy","y_val.npy","X_test.npy","y_test.npy","label_map.json"]
for f in files:
    fp = p / f
    if not fp.exists():
        print(f"MISSING: {fp}")
    else:
        if fp.suffix == ".npy":
            try:
                arr = np.load(fp)
                print(f"{f}: shape={arr.shape}, dtype={arr.dtype}, size_bytes={fp.stat().st_size}")
            except Exception as e:
                print(f"{f}: ERROR loading ({e})")
        elif fp.suffix == ".json":
            try:
                with open(fp) as fh:
                    d = json.load(fh)
                print(f"{f}: keys={list(d.keys())}, classes={len(d)}")
            except Exception as e:
                print(f"{f}: ERROR loading JSON ({e})")

# quick class distribution from CSVs (if present)
import csv
for split in ["train","val","test"]:
    csvp = p / f"{split}.csv"
    if csvp.exists():
        cnt={}
        with open(csvp) as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                lbl = r.get("label")
                cnt[lbl] = cnt.get(lbl,0)+1
        print(f"{split}.csv counts: {cnt}")
    else:
        print(f"{split}.csv missing")
