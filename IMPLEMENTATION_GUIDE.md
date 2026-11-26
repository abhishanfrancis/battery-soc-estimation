# Implementation guide

## 1. Read the measurements

`prepare.py` opens the publisher ZIP and loads each original Mixed drive-cycle MATLAB file. The raw structure provides `Time` in seconds, `TimeStamp` as wall-clock text, `Voltage` in volts, `Current` in amperes, `Battery_Temp_degC` in degrees Celsius and the cycler counter `Ah`. It also contains `Wh` and `Power`. The pipeline uses elapsed seconds to measure gaps. It selects the nearest real reading to each whole second, within 0.13 seconds. The EDA plots show one training experiment and training distributions.

## 2. Build the reference

Raw experiments have no SoC column. The publisher documents charging between tests at 1C to a 4.2 V limit and 50 mA cutoff, and its preparation notes describe anchoring discharges to 100% and dividing amp-hours by nominal capacity. We screen cycle starts for near-zero time and amp-hours and a plausible full-charge voltage. We calculate the reference as `1 + (Ah[t] - Ah[0]) / 3.0`, in fractional units. Negative current is discharge, as verified by the declining amp-hour counter and independently integrating signed current against elapsed time. The integrated counter and instrument counter must agree within 0.08 Ah. Small overshoot within one percentage point is clipped to [0, 1], and larger outliers are excluded. Across the 40 accepted files, the worst disagreement between the cycler counter and integrated current was 0.00550 Ah. Initial voltages ranged from 4.055 to 4.187 V, and the median raw step was approximately 0.1 s. The accepted experiments cover 261,003 samples after resampling, from 2,609,900 original measurements. The result is a nominal-capacity, coulomb-counted reference. Available capacity and empty-voltage behavior vary with temperature and load. [Publisher measurement protocol and preparation notes](https://data.mendeley.com/datasets/cp3473x7xv/3)

## 3. Keep the splits independent

Mixed profiles 1 through 5 train the models, profile 6 selects the weights, and profiles 7 and 8 form the test set. The profile number is held out across all temperatures, and intact experiments stay in one split. The same physical cell appears across the splits. Features are standardized using only training measurements. Neither Ah, time, ambient set point nor reference SoC is a model input. We exclude short cycles with less than 0.25 Ah net throughput to focus on substantial discharge experiments. See `data/experiments.csv` for included and skipped files.

## 4. Make 60-row windows

The features are ordered voltage, signed current and battery case temperature. Prediction target is the reference at the last row of the window. The code breaks acquisition blocks at invalid readings, non-increasing timestamps or raw gaps over 0.25 seconds. Each block has an ID that survives resampling. After resampling, steps outside 0.5 to 1.5 seconds also break windows. A window never spans these breaks or another experiment. Training takes a stride of 16 windows by default and validation/test take a stride of 5 to keep CPU runtime practical. All 60 individual measurements within a window are retained.

## 5. Train and compare

Both models use 64 recurrent units, 0.2 dropout, 32 recurrent units, 16 ReLU units and one sigmoid output. Adam minimizes MSE. Early stopping watches validation MSE with patience 4 and restores the best validation weights. The smoke run checks real data, both architectures and artifact generation in one epoch each. The benchmark writes MSE curves, reference and prediction plots, signed errors, predictions, timings and a comparison table.

Errors are in percentage points. MAE averages absolute errors. RMSE penalizes large errors more strongly. Maximum absolute error is the largest observed error among the evaluated windows. Training time includes Keras fit, validation and tracing. Inference time is a warmed-up batched pass on the local machine, including the Keras prediction loop and excluding preprocessing. Dividing that duration by the number of windows gives throughput-based milliseconds per window.

## 6. Use the saved models

`dashboard.py` applies the saved training scaler and the same windowing logic to the CSV. The first prediction in each continuous stretch appears on row 60. The app shows the latest estimate and a time series. The dashboard accepts measurement-only files and can export its predictions.

## Scope

This measures generalization to held-out drive-cycle profiles of a single lab-tested cell. The selected test set lacks 40 C cycles because those profiles failed the minimum-throughput rule. Results apply to the nominal-capacity reference and sampled windows. The lowest reference SoC in the accepted data is about 13.66%, so the benchmark does not cover empty-cell behavior. Broader claims about cell aging, new batteries or battery management deployment require separate experiments.
