# Intelligent Battery State of Charge Estimation

This repository organizes reproducible **LSTM** and **GRU** State-of-Charge (SoC) benchmarks across two lithium-ion battery datasets:

* **LG 18650HG2** ([lg/](lg/)) — Phillip Kollmeyer et al. (2020), Mendeley Data V3, DOI `10.17632/cp3473x7xv.3`
* **Panasonic NCR18650PF** ([panasonic/](panasonic/)) — Phillip Kollmeyer (2018), Mendeley Data V1, DOI `10.17632/wykht8y7tg.1`

Each model estimates present SoC from the latest **60 one-second readings** of `voltage_v`, `current_a`, and `temperature_c`.

## Repository Layout

```text
battery-soc-estimation/
├── common.py                 # Shared feature & sliding-window utilities
├── dashboard.py              # Unified Streamlit app (select LG or Panasonic, LSTM or GRU)
├── requirements.txt          # Python dependencies
├── BENCHMARK_COMPARISON.md   # Side-by-side metrics across both datasets
├── lg/                       # LG 18650HG2 pipeline, processed data, and trained models
│   ├── prepare.py
│   ├── train.py
│   ├── data/
│   └── results/
└── panasonic/                # Panasonic NCR18650PF pipeline, processed data, and trained models
    ├── prepare.py
    ├── preprocessing.py
    ├── training.py
    ├── lstm/                 # data/ and results/ for Panasonic LSTM
    └── gru/                  # data/ and results/ for Panasonic GRU
```

## Benchmark Summary

| Dataset | Model | Test MAE (pp) | Test RMSE (pp) | Max Abs Error (pp) | Test Windows | Batched Inference (ms/window) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **LG 18650HG2** | LSTM | 1.253 | 1.722 | 9.419 | 13,044 | 0.404 |
| **LG 18650HG2** | GRU | 1.322 | 1.786 | 9.788 | 13,044 | 0.273 |
| **Panasonic NCR18650PF** | LSTM | 1.528 | 2.028 | 10.648 | 7,681 | 0.170 |
| **Panasonic NCR18650PF** | GRU | 1.647 | 2.188 | 10.672 | 7,681 | 0.100 |

See [BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md) for full details.

## Run on Windows

Use a **Python 3.11 or 3.12** virtual environment (TensorFlow 2.19.1 does not support Python 3.13+):

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### Launch the Unified Dashboard

```powershell
.venv/Scripts/python.exe -m streamlit run dashboard.py
```

In the dashboard, select either **LG 18650HG2** or **Panasonic NCR18650PF**, choose **LSTM** or **GRU**, and run inference on the built-in held-out demo CSV or upload your own 1 Hz CSV (`time_s,voltage_v,current_a,temperature_c`).

### Re-run Training or Preprocessing (Optional)

All processed data and trained models are already included. To re-run from scratch:

```powershell
# LG 18650HG2
.venv/Scripts/python.exe lg/prepare.py
.venv/Scripts/python.exe lg/train.py

# Panasonic NCR18650PF
.venv/Scripts/python.exe panasonic/lstm/preprocess.py
.venv/Scripts/python.exe panasonic/lstm/train.py
.venv/Scripts/python.exe panasonic/gru/preprocess.py
.venv/Scripts/python.exe panasonic/gru/train.py
```
