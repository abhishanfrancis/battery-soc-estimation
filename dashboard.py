"""Try a saved model on a 1 Hz measurement CSV."""
import io

import joblib
import pandas as pd
import streamlit as st

from common import FEATURES, ROOT, TIME, windows

RESULTS = ROOT / "results"
DEMO = ROOT / "data" / "demo_measurements.csv"


@st.cache_resource
def load_artifacts(name, model_modified, scaler_modified):
    from tensorflow.keras.models import load_model

    model = load_model(RESULTS / f"{name.lower()}.keras", compile=False)
    return model, joblib.load(RESULTS / "scaler.joblib")


def infer(raw, model, scaler):
    frame = pd.read_csv(io.BytesIO(raw))
    missing = set([TIME, *FEATURES]) - set(frame)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    frame[[TIME, *FEATURES]] = frame[[TIME, *FEATURES]].apply(pd.to_numeric, errors="coerce")
    if "experiment_id" not in frame:
        frame["experiment_id"] = "Upload"
    elif frame["experiment_id"].isna().any():
        raise ValueError("Fill in every experiment_id or remove the column for a single experiment.")
    # Keep memory bounded for uploads. A chunk includes its preceding 59 rows.
    predictions = []
    for start in range(0, len(frame), 4096):
        prefix = max(0, start - 59)
        chunk = frame.iloc[prefix:start + 4096]
        x, endpoints = windows(chunk, scaler)
        if len(x):
            result = endpoints[[TIME, "experiment_id"]].copy()
            result["predicted_soc_pct"] = model.predict(x, batch_size=128, verbose=0).ravel() * 100
            predictions.append(result)
    if not predictions:
        raise ValueError("Provide at least 60 consecutive valid measurements within one experiment.")
    return frame, pd.concat(predictions, ignore_index=True)


def main():
    st.set_page_config(page_title="Battery SoC estimator")
    st.title("Battery State of Charge")
    st.markdown("Upload chronological **1 Hz** measurements with columns "
                "`time_s`, `voltage_v`, `current_a`, and `temperature_c`. "
                "Use negative current for discharge and positive for charge. "
                "Include `experiment_id` when the file contains multiple experiments.")
    choice = st.radio("Model", ["LSTM", "GRU"], horizontal=True)
    uploaded = st.file_uploader("Measurement CSV", type="csv")
    use_demo = st.checkbox("Use a real held-out dataset sample", value=DEMO.is_file())
    if uploaded is None and not (use_demo and DEMO.is_file()):
        return
    if not st.button("Estimate SoC", type="primary"):
        return
    model_path = RESULTS / f"{choice.lower()}.keras"
    scaler_path = RESULTS / "scaler.joblib"
    if not model_path.is_file() or not scaler_path.is_file():
        st.error("Run python train.py to build the saved model and scaler.")
        return
    try:
        with st.spinner("Estimating SoC"):
            model, scaler = load_artifacts(choice, model_path.stat().st_mtime, scaler_path.stat().st_mtime)
            frame, predictions = infer(uploaded.getvalue() if uploaded else DEMO.read_bytes(), model, scaler)
    except (ValueError, OSError) as error:
        st.error(str(error))
        return

    st.metric("Latest predicted SoC", f"{predictions['predicted_soc_pct'].iloc[-1]:.1f}%")
    st.caption(f"{len(predictions):,} predictions from {len(frame):,} input measurements")
    for column, label in zip(FEATURES, ["Voltage (V)", "Current (A)", "Battery case temperature (C)"]):
        st.line_chart(frame, x=TIME, y=column, color="experiment_id", y_label=label)
    st.line_chart(predictions, x=TIME, y="predicted_soc_pct", color="experiment_id", y_label="Predicted SoC (%)")
    st.download_button("Download predictions", predictions.to_csv(index=False), "soc_predictions.csv", "text/csv")


if __name__ == "__main__":
    main()
