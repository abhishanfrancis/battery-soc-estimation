# Benchmark Comparison: LG 18650HG2 vs Panasonic NCR18650PF

Both datasets use the same input features (`voltage_v`, `current_a`, `temperature_c`), 60-second sliding windows (`1 Hz`), and identical recurrent architectures:
* **LSTM** (`30,369` parameters): `LSTM(64, return_sequences=True) → Dropout(0.2) → LSTM(32) → Dense(16, relu) → Dense(1, sigmoid)`
* **GRU** (`23,201` parameters): `GRU(64, return_sequences=True) → Dropout(0.2) → GRU(32) → Dense(16, relu) → Dense(1, sigmoid)`

## Held-Out Test Results

| Dataset | Model | MAE (pp) | RMSE (pp) | Max Abs Error (pp) | Epochs (Best) | Train Time (s) | Inference (ms/window) | Test Windows | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LG 18650HG2 | LSTM | 1.253 | 1.722 | 9.419 | 15 (11) | 167.2 | 0.404 | 13,044 | [lg/results/comparison.csv](lg/results/comparison.csv) |
| LG 18650HG2 | GRU | 1.322 | 1.786 | 9.788 | 18 (16) | 194.6 | 0.273 | 13,044 | [lg/results/comparison.csv](lg/results/comparison.csv) |
| Panasonic NCR18650PF | LSTM | 1.528 | 2.028 | 10.648 | 17 (13) | 45.9 | 0.170 | 7,681 | [panasonic/lstm/results/comparison.csv](panasonic/lstm/results/comparison.csv) |
| Panasonic NCR18650PF | GRU | 1.647 | 2.188 | 10.672 | 18 (16) | 44.7 | 0.100 | 7,681 | [panasonic/gru/results/comparison.csv](panasonic/gru/results/comparison.csv) |
