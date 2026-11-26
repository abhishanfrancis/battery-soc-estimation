# Project progress

## Environment

Python 3.11.9 on native Windows. Intel Core i5-12450H, 8 physical cores, 12 logical processors, and 15.7 GiB of RAM. NVIDIA reports a GeForce RTX 3050 Laptop GPU with 4 GiB of VRAM. The project targets native Windows TensorFlow CPU and includes optional Linux/Colab GPU instructions. [TensorFlow platform support](https://www.tensorflow.org/install/pip)

## Data and preprocessing

- Inspected the original McMaster MATLAB files and the publisher's documentation.
- Converted 2,609,900 raw rows from 40 eligible experiments into 261,003 one-second measurements. Used real cycler amp-hours to build the nominal-capacity reference.
- Assigned the original profiles to 25 training, 5 validation and 10 test experiments. The manifest records all six exclusions and the reasons.
- Saved exploratory plots, the processed dataset, source provenance and a held-out CSV for the dashboard.

## Training and evaluation

- Installed the pinned packages in an isolated Python environment. Dependency checks passed, and TensorFlow successfully ran a CPU operation.
- Ran a one-epoch smoke test for both models, then trained the full benchmark. Saved both Keras models and the training-only scaler.
- Rebuilt preprocessing from the publisher ZIP using the pinned environment and confirmed the processed tables and source checksum match.
- Verified that windows stop at experiment changes, segment changes, timestamp gaps and invalid readings. Recomputed error metrics from saved predictions.
- Loaded both saved models in Streamlit tests. Each produced 1,741 predictions from the 1,800-row demo and rendered the latest estimate.
- See [Experimental results](EXPERIMENTAL_RESULTS.md) for measured metrics and the run settings.

## Attachments

The battery presentation informed the implementation. The attached PS146 Word document describes a fixed-asset depreciation project, which is outside this battery project's scope.
