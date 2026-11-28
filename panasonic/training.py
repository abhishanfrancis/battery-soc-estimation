"""Train one recurrent model on its own preprocessed Panasonic data."""
import argparse
import json
import os
from pathlib import Path
import platform
from shutil import copyfile
from time import perf_counter

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from common import FEATURES, TARGET, TIME, WINDOW

SEED = 146
BATCH = 128


def build(recurrent):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(WINDOW, len(FEATURES))),
        recurrent(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        recurrent(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def plots(history, predictions, name, destination):
    fig, ax = plt.subplots(figsize=(7, 4))
    for column, label in [("loss", "Train"), ("val_loss", "Validation")]:
        ax.plot(history["epoch"], history[column], "o-", label=label)
    ax.set(xlabel="Epoch", ylabel="MSE (SoC fraction squared)", title=f"{name} loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / f"{name.lower()}_loss.png", dpi=130)
    plt.close(fig)

    # Each held-out experiment gets its own axes so lines never join cycles.
    groups = list(predictions.groupby("experiment_id", sort=True))
    for chart in ("soc", "error"):
        rows = (len(groups) + 2) // 3
        fig, axes = plt.subplots(rows, 3, squeeze=False, figsize=(12, 2.4 * rows))
        for ax, (experiment, part) in zip(axes.flat, groups):
            if chart == "soc":
                ax.plot(part[TIME] / 60, part["reference_pct"], label="Reference", lw=1)
                ax.plot(part[TIME] / 60, part["prediction_pct"], label="Predicted", lw=1)
                ax.set_ylabel("SoC (%)")
                ax.set_ylim(0, 102)
            else:
                ax.axhline(0, color="gray", lw=0.7)
                ax.plot(part[TIME] / 60, part["error_pp"], lw=1)
                ax.set_ylabel("Error (pp)")
            ax.set(title=experiment, xlabel="Time (min)")
        if chart == "soc":
            axes[0, 0].legend(fontsize=8)
        for ax in list(axes.flat)[len(groups):]:
            ax.set_visible(False)
        fig.suptitle(f"{name} on held-out experiments")
        fig.tight_layout()
        fig.savefig(destination / f"{name.lower()}_{chart}.png", dpi=130)
        plt.close(fig)


def train(args, name):
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    if args.smoke:
        args.epochs = 1
    destination = args.out
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.data / "preprocessing.json").read_text())
    partitions = {}
    for split in ("train", "validation", "test"):
        x = np.load(args.data / f"{split}_windows.npy", allow_pickle=False)
        rows = pd.read_csv(args.data / f"{split}_endpoints.csv")
        if len(x) == 0 or len(x) != len(rows):
            raise ValueError(f"Missing or misaligned windows for {split}.")
        if args.smoke:
            x, rows = x[:128], rows.iloc[:128]
        partitions[split] = x, rows
    x_train, train_rows = partitions["train"]
    x_val, val_rows = partitions["validation"]
    x_test, test_rows = partitions["test"]
    run = dict(python=platform.python_version(), tensorflow=tf.__version__,
               devices=[d.name for d in tf.config.list_physical_devices()],
               seed=SEED, batch_size=BATCH, epochs_requested=args.epochs,
               threads=args.threads, train_stride=config["train_stride"], eval_stride=config["eval_stride"],
               smoke=args.smoke,
               splits={name: {"windows": len(x), "experiments": sorted(rows.experiment_id.unique())}
                       for name, (x, rows) in partitions.items()})
    (destination / "run.json").write_text(json.dumps(run, indent=2))
    print(run, flush=True)
    recurrent = {"LSTM": tf.keras.layers.LSTM, "GRU": tf.keras.layers.GRU}[name]
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = build(recurrent)
    stop = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
    start = perf_counter()
    fitted = model.fit(x_train, train_rows[TARGET].to_numpy(), batch_size=BATCH,
                       epochs=args.epochs, validation_data=(x_val, val_rows[TARGET].to_numpy()),
                       callbacks=[stop], verbose=2)
    training_seconds = perf_counter() - start
    model.predict(x_test[:BATCH], batch_size=BATCH, verbose=0)
    start = perf_counter()
    predicted = model.predict(x_test, batch_size=BATCH, verbose=0).ravel() * 100
    inference_seconds = perf_counter() - start
    predictions = test_rows[["experiment_id", TIME]].copy()
    predictions["reference_pct"] = test_rows[TARGET].to_numpy() * 100
    predictions["prediction_pct"] = predicted
    error = predictions["prediction_pct"] - predictions["reference_pct"]
    if not np.isfinite(error).all():
        raise ValueError(f"{name} produced non-finite predictions.")
    predictions["error_pp"] = error
    history = pd.DataFrame(fitted.history)
    history.insert(0, "epoch", np.arange(1, len(history) + 1))
    report = dict(model=name, mae_pp=abs(error).mean(), rmse_pp=np.sqrt(np.mean(error ** 2)),
                  max_abs_error_pp=abs(error).max(), train_seconds=training_seconds,
                  inference_seconds=inference_seconds,
                  inference_ms_per_window=1000 * inference_seconds / len(x_test),
                  parameters=model.count_params(), epochs=len(history),
                  best_epoch=int(history.loc[history["val_loss"].idxmin(), "epoch"]),
                  test_windows=len(x_test))
    model.save(destination / f"{name.lower()}.keras")
    copyfile(args.data / "scaler.joblib", destination / "scaler.joblib")
    predictions.to_csv(destination / f"{name.lower()}_predictions.csv", index=False)
    history.to_csv(destination / f"{name.lower()}_history.csv", index=False)
    plots(history, predictions, name, destination)
    print(report, flush=True)
    pd.DataFrame([report]).to_csv(destination / "comparison.csv", index=False)


def positive_int(text):
    result = int(text)
    if result < 1:
        raise argparse.ArgumentTypeError("provide a positive integer")
    return result


def main(name, directory):
    parser = argparse.ArgumentParser(description=f"Train the Panasonic {name} estimator.")
    parser.add_argument("--data", type=Path, default=directory / "data")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=positive_int, default=18)
    parser.add_argument("--threads", type=positive_int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.out is None:
        args.out = directory / ("smoke" if args.smoke else "results")
    train(args, name)
