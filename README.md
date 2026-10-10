# Bearing Condition Monitoring System

Synthetic proof-of-concept for vibration-based bearing condition monitoring using signal processing, machine learning, persistent alarm logic, and a Streamlit dashboard.

## Overview

The project demonstrates an end-to-end diagnostic pipeline:

```text
Vibration signal
      |
      v
Signal processing
      |
      v
Feature extraction
      |
      v
Machine-learning classification
      |
      v
Persistent alarm logic
      |
      v
Multi-machine monitoring
      |
      v
Streamlit dashboard
```

Current stage: **synthetic proof-of-concept**.

## Implemented

- synthetic vibration generation;
- bearing characteristic frequency calculation;
- FFT-based analysis;
- envelope analysis;
- 14 vibration features;
- five-class Random Forest classifier;
- model persistence with Joblib;
- reusable inference pipeline;
- multi-machine monitoring;
- persistent alarm logic;
- CSV monitoring history;
- Streamlit Hall Overview;
- Machine Details;
- Advanced Diagnostics;
- source abstraction through `SignalSource`.

## Planned

- live vibration sensor input;
- PLC / VFD integration;
- automatic RPM acquisition;
- industrial communication;
- machine-specific bearing configuration;
- real vibration data;
- model retraining and validation;
- production deployment.

## Detected Bearing Conditions

| Internal class | Meaning |
|---|---|
| `healthy` | Normal operation |
| `outer_race` | Outer race fault |
| `inner_race` | Inner race fault |
| `ball_fault` | Rolling element fault |
| `cage_fault` | Bearing cage fault |

## Bearing Frequencies

The project uses:

- **BPFO** — Ball Pass Frequency Outer Race
- **BPFI** — Ball Pass Frequency Inner Race
- **BSF** — Ball Spin Frequency
- **FTF** — Fundamental Train Frequency

For the current synthetic bearing geometry at 1800 RPM:

```text
Shaft frequency : 30 Hz
BPFO            : 96 Hz
BPFI            : 144 Hz
BSF             : 72 Hz
FTF             : 12 Hz
```

These frequencies change with shaft speed, so RPM is part of every measurement.

## Signal Processing

```text
Raw vibration signal
        |
        +--> Time-domain indicators
        |
        +--> FFT
        |
        +--> Band-pass filtering
        |
        +--> Hilbert transform
        |
        +--> Envelope analysis
        |
        v
14 extracted features
```

Current envelope band-pass range:

```text
800 Hz - 1600 Hz
```

## Extracted Features

### Time-domain
- `rms`
- `peak`
- `peak_to_peak`
- `crest_factor`
- `kurtosis`

### Raw spectral
- `shaft_amplitude`
- `bpfo_raw`
- `bpfi_raw`
- `bsf_raw`
- `ftf_raw`

### Envelope spectral
- `bpfo_envelope`
- `bpfi_envelope`
- `bsf_envelope`
- `ftf_envelope`

## Machine-Learning Model

```text
RandomForestClassifier
Trees        : 500
Random state : 42
Classes      : 5
```

Saved model:

```text
models/random_forest_multiclass_v3.joblib
```

## Synthetic Evaluation

Dataset:

```text
1500 samples
300 samples per class
5 classes
```

Cross-validation:

```text
Mean accuracy : 0.8567
Accuracy std  : 0.0128
Mean macro-F1 : 0.8612
Macro-F1 std  : 0.0125
```

Held-out synthetic test set:

```text
Accuracy : 0.8800
Macro-F1 : 0.8825
```

These are synthetic-data results only and must not be interpreted as industrial diagnostic accuracy.

## Feature Importance

| Feature | Approx. importance |
|---|---:|
| `bsf_envelope` | 0.2215 |
| `bpfi_envelope` | 0.1962 |
| `bpfo_envelope` | 0.1860 |
| `ftf_raw` | 0.1107 |
| `crest_factor` | 0.0283 |

## Inference Architecture

```text
SignalSource
      |
      v
VibrationMeasurement
      |
      v
extract_vibration_features()
      |
      v
BearingPredictor
      |
      v
Persistent alarm logic
```

Current source availability:

```text
Synthetic simulation  : ACTIVE
Live sensor input     : NOT IMPLEMENTED
PLC / VFD integration : NOT IMPLEMENTED
```

Every source returns a standard `VibrationMeasurement` containing:

```text
signal
rpm
sampling_frequency
timestamp
source_name
```

## Persistent Alarm Logic

The project separates:

- `prediction` — ML class;
- `model_state` — result from the current measurement window;
- `state` — confirmed persistent operational state.

Current configuration:

```text
FAULT confirmation  : 3 consecutive windows
CHECK confirmation  : 2 consecutive windows
NORMAL confirmation : 2 consecutive windows
FAULT clear         : 3 normal windows
```

Operator-facing states:

```text
🟢 NORMAL
🟡 CHECK
🔴 FAULT
```

## Multi-Machine Simulation

| Machine | Equipment | RPM |
|---|---|---:|
| M01 | Pump 01 | 1800 |
| M02 | Fan 02 | 1750 |
| M03 | Motor 03 | 1800 |
| M04 | Pump 04 | 1650 |
| M05 | Fan 05 | 1950 |

The software uses `machine_id`, so the architecture is not inherently limited to five machines.

## Monitoring History

Output file:

```text
data/processed/bearing_monitoring_history.csv
```

Important fields:

```text
machine_id
machine_name
timestamp
rpm
source_name
true_condition
prediction
confidence
model_state
state
consecutive_fault
consecutive_check
consecutive_normal
```

All 14 vibration features are also stored.

`true_condition` exists only in simulation because the simulator knows the injected condition.

## Dashboard

Run:

```powershell
uv run streamlit run app.py
```

### Hall Overview
- machines monitored;
- NORMAL / CHECK / FAULT counts;
- machine cards ordered by severity;
- machines requiring attention;
- hall-state distribution.

### Machine Details
- confirmed state;
- detected condition;
- confidence;
- RPM;
- latest analysis;
- confidence trend;
- vibration trend;
- monitoring history.

### Advanced Diagnostics
- BPFO / BPFI / BSF / FTF indicators;
- RMS;
- crest factor;
- kurtosis;
- persistence counters;
- feature history;
- acquisition information.

## RPM Acquisition in a Real Deployment

In the current proof-of-concept, RPM is defined by the simulator.

In a real installation, RPM should preferably be acquired automatically. Possible sources include:

- PLC;
- VFD;
- encoder;
- optical tachometer;
- inductive or proximity sensor.

PLC or VFD is the preferred direction when the speed signal is already available in the machine-control infrastructure. The final choice depends on the client.

## Real Deployment Concept

```text
Accelerometer
      |
      v
Vibration signal
      |
      +--------------------+
                           |
                       Edge PC
                           |
PLC / VFD ----------------+
     RPM                   |
                           v
               Bearing frequency calculation
                           |
                           v
                  Feature extraction
                           |
                           v
                    ML inference
                           |
                           v
                  Persistence logic
                           |
                           v
                      Dashboard
```

## Installation

Recommended Python version:

```text
Python 3.12
```

Install dependencies:

```powershell
uv sync
```

## Run the Monitoring Simulation

```powershell
uv run python src/monitor_bearing.py
```

This generates the monitoring history used by the dashboard.

## Run the Dashboard

```powershell
uv run streamlit run app.py
```

Default local address:

```text
http://localhost:8501
```

## Project Structure

```text
01_vibration_fault_detection/
│
├── app.py
├── README.md
├── pyproject.toml
├── uv.lock
│
├── data/
│   └── processed/
│       └── bearing_monitoring_history.csv
│
├── docs/
│   └── PROJECT_DOCUMENTATION.md
│
├── figures/
│
├── models/
│   └── random_forest_multiclass_v3.joblib
│
├── notebooks/
│
└── src/
    ├── bearing_predictor.py
    ├── monitor_bearing.py
    ├── predict_bearing_condition.py
    ├── predict_from_vibration_signal.py
    ├── signal_sources.py
    ├── synthetic_signal_generator.py
    ├── train_multiclass_model_v3.py
    ├── evaluate_multiclass_model_v3.py
    └── vibration_features.py
```

## Limitations

The current model is trained and demonstrated on synthetic signals.

Real machines can introduce:

- imbalance;
- misalignment;
- looseness;
- load variation;
- speed variation;
- gearbox effects;
- electrical noise;
- structural resonances;
- sensor mounting effects;
- machine-specific transfer paths.

A model trained only on synthetic signals may therefore behave differently on real machines because of **domain shift**.

## Industrial Validation

Before operational use, a real deployment would require:

1. calibrated sensors;
2. real vibration acquisition;
3. automatic RPM acquisition;
4. machine-specific bearing geometry;
5. representative operating data;
6. labelled fault examples where possible;
7. maintenance inspection comparison;
8. false-positive analysis;
9. false-negative analysis;
10. model retraining or adaptation;
11. persistence-threshold tuning;
12. long-duration validation.

## Disclaimer

This project is a **synthetic proof-of-concept and software architecture prototype**.

It has not been validated for industrial diagnostic accuracy and should not currently be used as a safety-critical machine protection or shutdown system.

## Next Development Phase

The next major phase is real acquisition while preserving the current downstream pipeline:

```text
SyntheticSignalSource
        |
        +------------------------+
                                 |
                         Diagnostic pipeline
                                 |
        +------------------------+
        |
RealSensorSource
```

The goal is to replace only the acquisition source while preserving feature extraction, machine-learning inference, persistence logic, monitoring storage, and the dashboard.

For the full technical explanation, see:

```text
docs/PROJECT_DOCUMENTATION.md
```
