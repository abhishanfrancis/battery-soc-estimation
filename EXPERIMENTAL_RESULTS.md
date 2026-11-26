# Experimental results

Both models were trained locally on the Intel Core i5-12450H CPU using Python 3.11.9 and TensorFlow 2.19.1. Each used the same seed (146), batch size (128), feature scaler, validation set and held-out test set.

| Split | Complete experiments | Windows |
| --- | ---: | ---: |
| Training | 25 | 10,150 |
| Validation | 5 | 6,237 |
| Test | 10 | 13,044 |

| Model | MAE (pp) | RMSE (pp) | Max absolute error (pp) | Fit time (s) | Inference (s) | Batched ms/window | Parameters | Epochs run | Best validation epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM | 1.253 | 1.722 | 9.419 | 167.18 | 5.275 | 0.404 | 30369 | 15 | 11 |
| GRU | 1.322 | 1.786 | 9.788 | 194.61 | 3.564 | 0.273 | 23201 | 18 | 16 |

LSTM had the lower MAE and RMSE. GRU used 7,168 fewer parameters and finished the timed prediction pass 1.71 seconds faster. These errors are measured against a nominal-capacity coulomb-counted reference. A single cell and one split support a baseline comparison for this dataset.

The test profiles were Mixed7 and Mixed8 at -20, -10, 0, 10 and 25 degrees Celsius. The fit budget was 18 epochs with patience 4. The train and evaluation strides were 16 and 5. The reported maximum is over the evaluated windows. Warmed batched inference includes the Keras prediction loop and excludes preprocessing. The per-window figure is total batched time divided by window count.

- [Comparison CSV](results/comparison.csv)
- [Run configuration](results/run.json)
- LSTM [loss](results/lstm_loss.png), [reference and predictions](results/lstm_soc.png), [signed errors](results/lstm_error.png)
- GRU [loss](results/gru_loss.png), [reference and predictions](results/gru_soc.png), [signed errors](results/gru_error.png)
- [Experiment manifest and exclusions](data/experiments.csv)
