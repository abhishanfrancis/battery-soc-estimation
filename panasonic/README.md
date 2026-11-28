# Panasonic NCR18650PF state-of-charge estimation

This project trains an LSTM and a GRU on Phillip Kollmeyer's measured 2.9 Ah Panasonic NCR18650PF drive cycles. Each estimator uses the most recent 60 one-second readings of voltage, signed current, and case temperature. The reference is computed as 1 + (Ah at the sample − Ah at the start) / 2.9, with the start anchored at full charge. It represents nominal-capacity coulomb counting.

## Run each pipeline

Install the dependencies in a Python 3.12 virtual environment.

```sh
python -m pip install -r requirements.txt
python lstm/preprocess.py
python lstm/train.py
python gru/preprocess.py
python gru/train.py
python -m streamlit run dashboard.py
```

Both preprocessing commands read the same publisher ZIP, which is downloaded to `raw/panasonic-source.zip` on first use. Pass `--archive PATH` to either preprocessing command to use an existing publisher ZIP.

Each command processes the raw MATLAB files, fits its own training-only scaler, and writes its measurements, scaler, manifest, plots, and train/validation/test windows into its model's `data/` directory. Each training command reads that model's files and writes its model, inference scaler, predictions, metrics, and plots into that model's `results/` directory. The dashboard uses the selected model's saved model and inference scaler.

```text
lstm/preprocess.py → lstm/data/ → lstm/train.py → lstm/results/
gru/preprocess.py  → gru/data/  → gru/train.py  → gru/results/
```

## Evaluation split

The 20 constant-chamber-temperature numbered drive cycles cover −20, −10, 0, 10 and 25 °C. Numbered cycles 1–2 train each model, cycle 3 selects weights, and cycle 4 tests each model. The numbered cycles remix some of the same underlying driving profiles. All experiments are from one cell. The benchmark evaluates new mixed recordings of that cell, with accuracy measured against the coulomb-counted reference.

| Split | Experiments | Windows per model |
| --- | ---: | ---: |
| Train | 10 | 4,641 |
| Validation | 5 | 6,961 |
| Test | 5 | 7,681 |

Dataset credit. Phillip Kollmeyer (2018), *Panasonic 18650PF Li-ion Battery Data*, Mendeley Data V1, DOI 10.17632/wykht8y7tg.1. [Dataset](https://data.mendeley.com/datasets/wykht8y7tg/1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Panasonic benchmark results

| Model | MAE (pp) | RMSE (pp) | Max absolute error (pp) | Batched inference (ms/window) |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 1.528 | 2.028 | 10.648 | 0.170 |
| GRU | 1.647 | 2.188 | 10.672 | 0.100 |

Errors are in SoC percentage points. Inference time reflects a warmed-up CPU batch and excludes preprocessing. All metrics use the same five held-out experiments. See each model's `results/comparison.csv` and `results/run.json` for exact metrics and run settings.
