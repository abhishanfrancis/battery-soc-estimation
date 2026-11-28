from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES = ["voltage_v", "current_a", "temperature_c"]
TIME = "time_s"
TARGET = "reference_soc"
WINDOW = 60
GAP = 1.5


def continuous_blocks(frame, fields, max_gap=GAP, min_step=0.5):
    """Keep acquisition order and break at missing, repeated or gapped timestamps."""
    block_start = None
    values = frame[[TIME, *fields]].to_numpy(dtype=float)
    good = np.isfinite(values).all(axis=1)
    good &= frame["voltage_v"].between(2, 4.35).to_numpy()
    good &= frame["current_a"].between(-75, 75).to_numpy()
    good &= frame["temperature_c"].between(-40, 80).to_numpy()
    for index, valid in enumerate(good):
        delta = values[index, 0] - values[index - 1, 0] if index else 0
        if block_start is not None and (not valid or not min_step <= delta <= max_gap or delta == 0):
            yield frame.iloc[block_start:index]
            block_start = None
        if valid and block_start is None:
            block_start = index
    if block_start is not None:
        yield frame.iloc[block_start:]


def windows(frame, scaler, stride=1, labeled=False):
    arrays = []
    endpoints = []
    fields = FEATURES + ([TARGET] if labeled else [])
    # An upload may contain several contiguous experiments.
    groups = [frame]
    identifiers = [field for field in ("experiment_id", "segment_id") if field in frame]
    if identifiers:
        run = frame[identifiers].ne(frame[identifiers].shift()).any(axis=1).cumsum()
        groups = (part for _, part in frame.groupby(run, sort=False))
    for group in groups:
        for block in continuous_blocks(group, fields):
            if len(block) < WINDOW:
                continue
            scaled = scaler.transform(block[FEATURES]).astype("float32")
            view = np.lib.stride_tricks.sliding_window_view(scaled, WINDOW, axis=0)
            arrays.append(view[::stride].transpose(0, 2, 1).copy())
            endpoints.append(block.iloc[WINDOW - 1::stride])
    if not arrays:
        return np.empty((0, WINDOW, 3), dtype="float32"), frame.iloc[:0]
    return np.concatenate(arrays), pd.concat(endpoints, ignore_index=True)
