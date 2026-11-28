"""Process the publisher archive and fit one model's feature pipeline."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import FEATURES, ROOT, TARGET, TIME, windows
from prepare import download, prepare

STRIDES = {"train": 16, "validation": 5, "test": 5}


def main(directory):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "raw" / "panasonic-source.zip")
    args = parser.parse_args()
    if not args.archive.exists():
        download(args.archive)
    destination = directory / "data"
    prepare(args.archive, destination)
    measurements = pd.read_csv(destination / "measurements.csv.gz")
    if not measurements.groupby("experiment_id").split.nunique().eq(1).all():
        raise ValueError("An experiment appears in multiple splits.")
    scaler = StandardScaler().fit(measurements.loc[measurements.split.eq("train"), FEATURES])
    for split, stride in STRIDES.items():
        x, endpoints = windows(measurements.loc[measurements.split.eq(split)], scaler,
                               stride, labeled=True)
        if not len(x):
            raise ValueError(f"No usable windows for {split}.")
        np.save(destination / f"{split}_windows.npy", x, allow_pickle=False)
        endpoints[["experiment_id", TIME, TARGET]].to_csv(
            destination / f"{split}_endpoints.csv", index=False
        )
        print(f"{split}: {len(x):,} windows from {endpoints.experiment_id.nunique()} experiments")
    joblib.dump(scaler, destination / "scaler.joblib")
    settings = {"train_stride": STRIDES["train"], "eval_stride": STRIDES["test"]}
    (destination / "preprocessing.json").write_text(json.dumps(settings, indent=2))
