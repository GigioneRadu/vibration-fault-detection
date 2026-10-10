# Bearing Condition Monitoring System
## Synthetic Proof-of-Concept for Vibration-Based Predictive Maintenance

---

## 1. Project Overview

This project implements an end-to-end proof-of-concept for vibration-based bearing condition monitoring.

The current system demonstrates the complete software pipeline required to move from a vibration signal to an operational machine state:

1. generate or acquire a vibration signal;
2. use the machine shaft speed (RPM);
3. calculate characteristic bearing frequencies;
4. extract vibration features;
5. classify the bearing condition with a machine-learning model;
6. apply persistent alarm logic across consecutive measurement windows;
7. monitor multiple machines;
8. display the results in an operator-oriented dashboard.

The current implementation uses **synthetic vibration signals**. It is therefore a software and signal-processing proof-of-concept, not an industrially validated diagnostic system.

---

## 2. Current Scope

The current version implements:

- five monitored machines;
- synthetic vibration signal generation;
- shaft-speed-dependent bearing frequency calculation;
- FFT-based spectral analysis;
- envelope analysis using a Hilbert transform;
- extraction of 14 vibration features;
- Random Forest multiclass classification;
- five bearing-condition classes;
- persistent alarm confirmation logic;
- multi-machine monitoring history;
- Streamlit dashboard;
- operator-oriented and technical diagnostic views.

The following items are planned for a future real deployment but are **not yet implemented**:

- live accelerometer acquisition;
- PLC/VFD communication;
- automatic RPM acquisition from industrial control equipment;
- industrial network transport;
- real sensor calibration;
- industrial validation on labelled real vibration data;
- production alarm integration.

---

## 3. Bearing Conditions

The machine-learning model currently distinguishes five classes:

| Internal class | User-facing meaning |
|---|---|
| `healthy` | Normal operation |
| `outer_race` | Outer race fault |
| `inner_race` | Inner race fault |
| `ball_fault` | Rolling element fault |
| `cage_fault` | Bearing cage fault |

---

## 4. Bearing Geometry

The synthetic proof-of-concept uses:

- 8 rolling elements;
- rolling-element diameter: 8 mm;
- pitch diameter: 40 mm;
- contact angle: 0°.

The shaft rotational frequency is:

`shaft_frequency = RPM / 60`

At 1800 RPM:

- shaft frequency: 30 Hz;
- BPFO: 96 Hz;
- BPFI: 144 Hz;
- BSF: 72 Hz;
- FTF: 12 Hz.

---

## 5. Characteristic Bearing Frequencies

### BPFO
Ball Pass Frequency Outer Race — mainly associated with defects on the outer race.

### BPFI
Ball Pass Frequency Inner Race — mainly associated with defects on the inner race.

### BSF
Ball Spin Frequency — mainly associated with rolling-element defects.

### FTF
Fundamental Train Frequency — mainly associated with bearing-cage defects.

These frequencies depend on shaft speed and bearing geometry, so the current RPM must be known before frequency-dependent features are extracted.

---

## 6. Signal Processing Pipeline

```text
Vibration signal
      |
      v
Time-domain indicators
      |
      +--> RMS
      +--> Peak
      +--> Peak-to-peak
      +--> Crest factor
      +--> Kurtosis
      |
      v
FFT
      |
      +--> Shaft amplitude
      +--> BPFO raw amplitude
      +--> BPFI raw amplitude
      +--> BSF raw amplitude
      +--> FTF raw amplitude
      |
      v
Band-pass filter: 800-1600 Hz
      |
      v
Hilbert transform
      |
      v
Envelope signal
      |
      v
Envelope FFT
      |
      +--> BPFO envelope
      +--> BPFI envelope
      +--> BSF envelope
      +--> FTF envelope
```

---

## 7. Envelope Analysis

Bearing faults often generate short impacts rather than clean low-frequency sinusoidal signals.

These impacts can excite a higher-frequency machine resonance. The project therefore:

1. band-pass filters the signal between 800 and 1600 Hz;
2. applies the Hilbert transform;
3. calculates the amplitude envelope;
4. removes the envelope mean;
5. calculates the envelope spectrum;
6. evaluates BPFO, BPFI, BSF and FTF amplitudes.

---

## 8. Extracted Features

The model receives 14 features.

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

---

## 9. Synthetic Signal Generation

The current signal generator includes:

- shaft rotation;
- second shaft harmonic;
- random vibration noise;
- fault-related periodic impacts;
- high-frequency resonance excitation;
- timing jitter;
- characteristic fault frequencies.

Typical simulation parameters:

- sampling frequency: 5000 Hz;
- window duration: 1 second;
- resonance frequency: approximately 1200 Hz.

The synthetic generator exists only for development, testing and demonstration until a real acquisition source is connected.

---

## 10. Machine-Learning Model

Current classifier:

- `RandomForestClassifier`;
- 500 trees;
- `random_state=42`;
- five classes.

Saved model:

`models/random_forest_multiclass_v3.joblib`

---

## 11. Synthetic Evaluation Results

Final V3 dataset:

- 1500 samples;
- 300 samples per class;
- 5 classes.

Cross-validation:

- mean accuracy: 0.8567;
- standard deviation: 0.0128;
- mean macro-F1: 0.8612;
- standard deviation: 0.0125.

Held-out synthetic test set:

- accuracy: 0.8800;
- macro-F1: 0.8825.

Approximate class results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Ball fault | 0.95 | 0.92 | 0.93 |
| Cage fault | 0.84 | 0.72 | 0.77 |
| Healthy | 0.69 | 0.88 | 0.77 |
| Inner race fault | 0.98 | 0.93 | 0.96 |
| Outer race fault | 1.00 | 0.95 | 0.97 |

These results demonstrate internal consistency of the synthetic model only. They are **not industrial diagnostic accuracy figures**.

---

## 12. Feature Importance

Approximate permutation importance:

| Feature | Importance |
|---|---:|
| `bsf_envelope` | 0.2215 |
| `bpfi_envelope` | 0.1962 |
| `bpfo_envelope` | 0.1860 |
| `ftf_raw` | 0.1107 |
| `crest_factor` | 0.0283 |

The strongest features correspond well to the simulated physical fault signatures.

---

## 13. Inference Pipeline

### `src/vibration_features.py`
Responsible for:

- bearing geometry;
- bearing-frequency calculation;
- FFT amplitudes;
- envelope calculation;
- 14-feature extraction.

### `src/bearing_predictor.py`
Responsible for:

- model loading;
- correct feature order;
- class probabilities;
- single-window state.

Single-window logic:

```text
confidence < 0.60
    -> CHECK

healthy and confidence >= 0.60
    -> NORMAL

fault class and confidence >= 0.60
    -> FAULT
```

This single-window result is stored as `model_state`.

---

## 14. Persistent Alarm Logic

The system distinguishes:

- `prediction` = ML class;
- `model_state` = result for the current one-second window;
- `state` = persistent confirmed machine state.

Current settings:

```text
FAULT_CONFIRMATION_WINDOWS = 3
NORMAL_CONFIRMATION_WINDOWS = 2
CHECK_CONFIRMATION_WINDOWS = 2
FAULT_CLEAR_WINDOWS = 3
```

A confirmed fault therefore requires three consecutive FAULT windows.

A single uncertain reading does not immediately downgrade a stable machine.

A confirmed FAULT is not cleared by one healthy window.

---

## 15. Multi-Machine Monitoring

Current demonstration:

| Machine | Equipment | RPM |
|---|---|---:|
| M01 | Pump 01 | 1800 |
| M02 | Fan 02 | 1750 |
| M03 | Motor 03 | 1800 |
| M04 | Pump 04 | 1650 |
| M05 | Fan 05 | 1950 |

The software architecture uses `machine_id`, so it is not inherently limited to five machines.

---

## 16. Monitoring History

Important stored fields:

```text
machine_id
machine_name
timestamp
rpm
prediction
confidence
model_state
state
consecutive_fault
consecutive_check
consecutive_normal
```

All 14 vibration features are also stored.

The simulator additionally stores `true_condition`. This field would not normally exist in a real live system because the true bearing condition is not automatically known.

---

## 17. Dashboard

### Hall Overview

Designed primarily for operators and maintenance supervision.

It shows:

- machines monitored;
- NORMAL count;
- CHECK count;
- FAULT count;
- machine cards ordered by severity;
- machines requiring attention;
- hall-state distribution.

### Machine Details

Shows:

- confirmed state;
- detected condition;
- confidence;
- RPM;
- latest analysis;
- confidence trend;
- vibration-level trend;
- monitoring history.

### Advanced diagnostics

Contains technical indicators:

- BPFO envelope;
- BPFI envelope;
- BSF envelope;
- FTF raw;
- RMS;
- crest factor;
- kurtosis;
- persistence counters;
- complete feature history.

The intention is to keep the main interface understandable to operators while retaining detailed diagnostics for maintenance or engineering personnel.

---

## 18. Real Deployment Architecture

```text
                    MACHINE
                       |
             +---------+---------+
             |                   |
             v                   v
      Accelerometer          PLC / VFD
      vibration data             |
             |                Current RPM
             +---------+---------+
                       |
                       v
              Edge PC / mini-PC
                       |
                       v
              Signal preprocessing
                       |
                       v
        Bearing frequency calculation
          BPFO / BPFI / BSF / FTF
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
             Storage / Dashboard
```

---

## 19. RPM Acquisition in a Real Deployment

In the proof-of-concept, RPM values are defined in the simulator.

In a real system, RPM should ideally **not be entered manually by the operator**.

Preferred sources:

1. PLC;
2. VFD;
3. encoder;
4. optical tachometer;
5. inductive/proximity speed sensor.

If the PLC or VFD already exposes machine speed, that signal should normally be reused rather than duplicated with additional hardware.

---

## 20. Why RPM Must Be Automatic

Bearing characteristic frequencies move with shaft speed.

Example:

```text
1800 RPM -> 30 Hz shaft frequency
1500 RPM -> 25 Hz shaft frequency
```

BPFO, BPFI, BSF and FTF therefore also change.

The intended real pipeline is conceptually:

```python
rpm = read_rpm_from_plc()

features = extract_vibration_features(
    signal=vibration_signal,
    rpm=rpm,
    sampling_frequency=5000,
)
```

The system recalculates the characteristic bearing frequencies from the current RPM before feature extraction.

---

## 21. PLC / VFD Integration

Possible future industrial interfaces include:

- OPC UA;
- Modbus TCP;
- Profinet;
- EtherNet/IP;
- vendor-specific PLC interfaces.

The actual protocol depends on the plant infrastructure.

These protocols are **not implemented in the current project**.

---

## 22. Planned Real Sensor Interface

The next architectural change will separate the data source from the diagnostic pipeline.

Planned design:

```text
SignalSource
    |
    +--> SyntheticSignalSource
    |
    +--> RealSensorSource
```

The downstream processing should receive the same vibration-window format regardless of source.

This allows the existing feature extraction, model inference and persistence logic to remain unchanged.

---

## 23. Planned UI Representation

Until real acquisition is implemented, unavailable sources should appear disabled:

```text
Data source

● Synthetic simulation
○ Live sensor input        [Coming soon]
○ PLC / VFD integration    [Coming soon]
```

This makes the planned architecture visible without implying that unfinished hardware integration already works.

---

## 24. Real Sensor Considerations

A real acquisition design must define:

- accelerometer type;
- mounting method;
- measurement axis;
- sensor sensitivity;
- sampling rate;
- ADC resolution;
- anti-alias filtering;
- noise and grounding;
- calibration;
- synchronization between vibration and RPM.

These factors directly influence diagnostic quality.

---

## 25. Edge Computing

A possible edge workflow is:

```text
Acquire vibration
      |
Acquire RPM
      |
Build measurement window
      |
Extract features
      |
Run inference
      |
Update persistent state
      |
Send monitoring record
```

This reduces network traffic because the central system does not necessarily need every raw high-frequency sample continuously.

Raw vibration can be retained selectively for alarms, troubleshooting or engineering analysis.

---

## 26. Limitations

### Synthetic data
The current model is trained and demonstrated on synthetic signals.

### Domain shift
Real machines can differ significantly from the simulated signal distribution.

### No industrial validation
No labelled industrial dataset or controlled fault rig has yet been used to validate the model.

### No safety certification
The project must not currently be used as a safety-critical shutdown system.

### Fixed bearing geometry
The present proof-of-concept assumes one configured bearing geometry. A real system would need machine-specific bearing configuration.

---

## 27. Industrial Validation Requirements

A future industrial validation phase would require:

1. calibrated sensor installation;
2. automatic RPM acquisition;
3. real vibration recordings;
4. data across varying operating conditions;
5. correct bearing geometry for each asset;
6. labelled fault examples where possible;
7. comparison with maintenance inspections;
8. false-positive and false-negative analysis;
9. retraining or model adaptation using real data;
10. persistence-threshold tuning;
11. plant-specific alarm rules;
12. long-duration monitoring trials.

---

## 28. Project Status

### Implemented

- bearing-frequency calculation;
- synthetic vibration generation;
- FFT analysis;
- envelope analysis;
- 14-feature extraction;
- five-class Random Forest;
- synthetic evaluation;
- Joblib model persistence;
- reusable predictor;
- multi-machine monitoring;
- persistent alarm logic;
- CSV history;
- Hall Overview;
- Machine Details;
- Advanced Diagnostics.

### Planned

- common signal-source interface;
- real vibration source;
- PLC/VFD RPM input;
- industrial communication;
- machine/bearing configuration database;
- real-data training and validation;
- production deployment architecture.

---

## 29. Next Development Step

The next software step is the introduction of a common signal-source interface.

The synthetic source will remain active for testing and demonstration.

The real sensor source will initially be present in the architecture but disabled in the dashboard until acquisition is actually implemented.

Only after that will PLC/VFD RPM integration and real sensor testing be connected to the existing diagnostic pipeline.

---

## 30. Conclusion

The current project demonstrates the complete software path:

```text
Synthetic vibration
        |
        v
Signal processing
        |
        v
Feature extraction
        |
        v
Machine learning
        |
        v
Persistent machine state
        |
        v
Multi-machine dashboard
```

The target architecture becomes:

```text
Real accelerometer + PLC/VFD RPM
              |
              v
      Signal processing
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

The project should therefore be presented as a **synthetic proof-of-concept and software architecture prototype**, with the next phase focused on replaceable real-data acquisition and controlled validation.
