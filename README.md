# Intelligent Battery State of Charge Estimation

This project compares LSTM and GRU on measured LG 18650HG2 mixed drive cycles. Each model estimates present state of charge from the latest 60 one-second voltage, current and battery case temperature readings.

## Run on Windows

Open PowerShell in this project directory. The delivered benchmark was trained on the native Windows CPU because TensorFlow 2.19 does not expose CUDA on native Windows. [TensorFlow installation guide](https://www.tensorflow.org/install/pip) Use Python 3.11 and a virtual environment. Using its Python executable directly avoids PowerShell activation policy issues.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices())"
```

### Try the dashboard

```powershell
.venv/Scripts/python.exe -m streamlit run dashboard.py
```

Open the local URL printed by Streamlit. Choose either model and click **Estimate SoC**. The built-in demo uses real measurements from a held-out experiment. You can also upload a CSV with `time_s,voltage_v,current_a,temperature_c` and optional `experiment_id,segment_id`. Keep rows in recording order, sampled at approximately one second, with discharge current negative and charging current positive. When starting a new experiment or recording segment, change its ID. Download predictions from the dashboard.

### Rebuild the experiment

The delivered processed data can be used immediately for training. To rebuild it from raw source measurements, run `prepare.py`. The automatic download is several hundred MB. Alternatively click **Download All** on the version 3 dataset page, save the ZIP without extracting it, and use `prepare.py --archive "C:/path/to/archive.zip"`.

```powershell
.venv/Scripts/python.exe prepare.py
.venv/Scripts/python.exe train.py --smoke
.venv/Scripts/python.exe train.py
```

The smoke run uses small slices and writes to `smoke/`. The benchmark uses all eligible experiments and writes both models, scaler, plots, CSVs and run settings to `results/`. To keep the delivered benchmark unchanged during an experiment, pass `--out results-new`. The dashboard loads artifacts from `results/`.

### Google Colab

Only needed if you want a faster or denser training run. Upload and unzip the project or copy it from Drive, change to its directory, and use the following in a notebook. Select a GPU runtime before training if one is available. The processed data included here lets you skip the raw download.

```python
%cd /content/battery_soc
%pip install -r requirements.txt "tensorflow[and-cuda]==2.19.1"
```

Restart the Colab session after installation if requested. Check TensorFlow's device list and run the same scripts.

```python
!python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
!python train.py --smoke
!python train.py --epochs 30 --train-stride 4 --out colab-results
```

### Data credit

Philip Kollmeyer, Carlos Vidal, Mina Naguib and Michael Skells (2020), *LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC Estimator Script*, Mendeley Data V3, DOI 10.17632/cp3473x7xv.3, CC BY 4.0. This project uses the raw mixed-cycle MAT files and creates its own 1 Hz samples and nominal-capacity references. See the experiment manifest for source filenames. [Dataset, original documentation and download](https://data.mendeley.com/datasets/cp3473x7xv/3) ? [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)
