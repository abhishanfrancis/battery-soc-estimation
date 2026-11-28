"""Read raw McMaster drive cycles and preserve their acquisition boundaries."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from common import FEATURES, ROOT, TARGET, TIME, continuous_blocks

URL = "https://data.mendeley.com/public-api/zip/cp3473x7xv/download/3"
CAPACITY_AH = 3.0
SOURCE_FIELDS = ["Voltage", "Current", "Battery_Temp_degC"]


def download(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    with urlopen(URL, timeout=180) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    with ZipFile(temporary):
        pass
    temporary.replace(destination)


def read_cycle(archive, member):
    document = loadmat(io.BytesIO(archive.read(member)), simplify_cells=True)
    raw = next(value for key, value in document.items() if not key.startswith("_"))
    time = np.asarray(raw["Time"], dtype=float)
    current = np.asarray(raw["Current"], dtype=float)
    ah = np.asarray(raw["Ah"], dtype=float)

    # Reject questionable full-charge starts and verify cycler Ah against current.
    if not (abs(time[0]) <= 1 and abs(ah[0]) <= 0.02 and 4.05 <= raw["Voltage"][0] <= 4.25):
        raise ValueError("start did not pass full-charge checks")
    dt = np.diff(time)
    integrated = np.r_[0, np.cumsum((current[:-1] + current[1:]) * dt / 7200)]
    relative_ah = ah - ah[0]
    valid = np.isfinite(integrated) & np.isfinite(relative_ah)
    if not valid.any() or not np.isfinite(relative_ah[-1]) or relative_ah[-1] >= 0:
        raise ValueError("missing or inconsistent discharge counter")
    if relative_ah[-1] > -0.25:
        raise ValueError("short discharge with less than 0.25 Ah net throughput")
    check_ah = float(np.max(np.abs(integrated[valid] - relative_ah[valid])))
    if check_ah > 0.08:
        raise ValueError(f"current integration disagrees with cycler Ah by {check_ah:.3f} Ah")

    frame = pd.DataFrame({TIME: time, **dict(zip(FEATURES, (raw[k] for k in SOURCE_FIELDS)))})
    soc = 1 + relative_ah / CAPACITY_AH
    frame[TARGET] = np.where((soc >= -0.01) & (soc <= 1.01), soc.clip(0, 1), np.nan)
    samples = []
    for segment_id, block in enumerate(continuous_blocks(frame, FEATURES + [TARGET], max_gap=0.25, min_step=0)):
        # Keep the nearest recorded sample to each whole second, without filling gaps.
        t = block[TIME].to_numpy()
        grid = np.arange(np.ceil(t[0]), np.floor(t[-1]) + 1)
        right = np.searchsorted(t, grid)
        left = np.maximum(0, right - 1)
        nearest = np.where(grid - t[left] < t[right] - grid, left, right)
        selection = abs(t[nearest] - grid) <= 0.13
        sampled = block.iloc[nearest[selection]].copy()
        sampled[TIME] = grid[selection]
        sampled["segment_id"] = segment_id
        samples.append(sampled)
    if not samples:
        raise ValueError("no usable continuous samples")
    sampled = pd.concat(samples, ignore_index=True)
    if not any(len(block) >= 60 for _, block in sampled.groupby("segment_id")):
        raise ValueError("no continuous 60-sample sequence")
    report = dict(raw_rows=len(frame), samples_1hz=len(sampled), raw_dt_median_s=np.median(dt),
                  positive_current_a=np.nanmax(current), negative_current_a=np.nanmin(current),
                  start_voltage_v=raw["Voltage"][0], net_ah=relative_ah[-1], max_integration_delta_ah=check_ah,
                  min_reference_pct=100 * sampled[TARGET].min(), max_reference_pct=100 * sampled[TARGET].max())
    return sampled, report


def eda(data, manifest, destination):
    cycle = manifest.query("split == 'train'").sort_values("samples_1hz", ascending=False).iloc[0]
    frame = data.query("experiment_id == @cycle.experiment_id")
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, field, units in zip(axes, FEATURES + [TARGET], ["Voltage (V)", "Current (A)", "Case temp (C)", "Reference SoC (%)"]):
        for block in continuous_blocks(frame, FEATURES + [TARGET]):
            ax.plot(block[TIME], block[field] * (100 if field == TARGET else 1), lw=1, color="tab:blue")
        ax.set_ylabel(units)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Experiment time (s)")
    fig.suptitle(f"Training experiment {cycle.experiment_id}")
    fig.tight_layout()
    fig.savefig(destination / "eda_timeseries.png", dpi=130)
    plt.close(fig)

    train = data[data["split"] == "train"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, field in zip(axes, FEATURES + [TARGET]):
        ax.hist(train[field] * (100 if field == TARGET else 1), bins=40)
        ax.set_xlabel("Reference SoC (%)" if field == TARGET else field)
    axes[0].set_ylabel("Training samples")
    fig.tight_layout()
    fig.savefig(destination / "eda_distributions.png", dpi=130)
    plt.close(fig)


def prepare(path, destination):
    tables, report = [], []
    with ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            name = PurePosixPath(member)
            match = re.search(r"_(?:Mixed|Mix)([1-8])_.*\.mat$", name.name, re.I)
            if not match or "Original_Dataset" not in member:
                continue
            number = int(match[1])
            temperature = int(name.parent.name.replace("degC", "").replace("n", "-"))
            split = "test" if number >= 7 else "validation" if number == 6 else "train"
            experiment = f"{temperature:+03d}C_Mixed{number}"
            record = dict(experiment_id=experiment, ambient_c=temperature, split=split, source=member)
            try:
                frame, stats = read_cycle(archive, member)
            except ValueError as error:
                report.append({**record, "status": str(error)})
                print(f"{experiment} skipped: {error}", flush=True)
                continue
            tables.append(frame.assign(experiment_id=experiment, split=split))
            report.append({**record, **stats, "status": "included"})
    manifest = pd.DataFrame(report)
    if not tables or set(manifest.query("status == 'included'").split) != {"train", "validation", "test"}:
        raise ValueError("Archive must contain usable raw Mixed drive cycles for all three splits.")
    destination.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    (destination / "source.json").write_text(json.dumps({"url": URL, "sha256": digest}, indent=2))
    data = pd.concat(tables, ignore_index=True)
    data.to_csv(destination / "measurements.csv.gz", index=False)
    manifest.to_csv(destination / "experiments.csv", index=False)
    eda(data, manifest, destination)
    demo = next(part for _, part in data.query("split == 'test'").groupby("experiment_id") if len(part) > 600)
    demo.iloc[:1800][[TIME, *FEATURES, "experiment_id", "segment_id"]].to_csv(destination / "demo_measurements.csv", index=False)
    print(manifest.groupby(["status", "split"]).size().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "hg2.zip")
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if not args.archive.exists():
        download(args.archive)
    prepare(args.archive, args.out)
