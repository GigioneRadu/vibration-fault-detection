import pandas as pd

from bearing_predictor import BearingPredictor
from synthetic_signal_generator import (
    generate_synthetic_vibration_signal,
)
from vibration_features import (
    extract_vibration_features,
)


# ============================================================
# TEST SETTINGS
# ============================================================

SAMPLING_FREQUENCY = 5000
DURATION = 1.0
RPM = 1800.0


# ============================================================
# CONDITIONS TO TEST
# ============================================================

TEST_CONDITIONS = [
    "healthy",
    "outer_race",
    "inner_race",
    "ball_fault",
    "cage_fault",
]


# ============================================================
# CREATE PREDICTOR
# ============================================================

predictor = BearingPredictor(
    confidence_threshold=0.60
)


# ============================================================
# RUN TEST PIPELINE
# ============================================================

results = []

for condition in TEST_CONDITIONS:

    # --------------------------------------------------------
    # 1. GENERATE RAW SIGNAL
    # --------------------------------------------------------

    raw_signal = generate_synthetic_vibration_signal(
        fault_type=condition,
        rpm=RPM,
        sampling_frequency=SAMPLING_FREQUENCY,
        duration=DURATION,
        random_seed=12345,
    )


    # --------------------------------------------------------
    # 2. EXTRACT FEATURES
    # --------------------------------------------------------

    features = extract_vibration_features(
        signal=raw_signal,
        rpm=RPM,
        sampling_frequency=SAMPLING_FREQUENCY,
    )


    # --------------------------------------------------------
    # 3. MODEL INFERENCE
    # --------------------------------------------------------

    prediction = predictor.predict(
        features
    )


    # --------------------------------------------------------
    # 4. STORE RESULT
    # --------------------------------------------------------

    results.append(
        {
            "true_condition":
                condition,

            "prediction":
                prediction[
                    "predicted_class"
                ],

            "confidence":
                prediction[
                    "confidence"
                ],

            "state":
                prediction[
                    "condition_state"
                ],

            "bpfo_envelope":
                features[
                    "bpfo_envelope"
                ],

            "bpfi_envelope":
                features[
                    "bpfi_envelope"
                ],

            "bsf_envelope":
                features[
                    "bsf_envelope"
                ],

            "ftf_raw":
                features[
                    "ftf_raw"
                ],
        }
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# MAIN RESULTS
# ============================================================

print("=" * 120)

print(
    "MODULAR VIBRATION CONDITION MONITORING PIPELINE"
)

print("=" * 120)

print()

print(
    results_df[
        [
            "true_condition",
            "prediction",
            "confidence",
            "state",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# DIAGNOSTIC FEATURES
# ============================================================

print()

print("=" * 120)

print(
    "DIAGNOSTIC FEATURES"
)

print("=" * 120)

print()

print(
    results_df[
        [
            "true_condition",
            "bpfo_envelope",
            "bpfi_envelope",
            "bsf_envelope",
            "ftf_raw",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SIMPLE TEST ACCURACY
# ============================================================

correct_predictions = (
    results_df[
        "true_condition"
    ]
    == results_df[
        "prediction"
    ]
).sum()

total_predictions = len(
    results_df
)


print()

print("=" * 120)

print(
    f"Correct predictions: "
    f"{correct_predictions}/"
    f"{total_predictions}"
)

print(
    f"Simple test accuracy: "
    f"{correct_predictions / total_predictions:.2%}"
)

print("=" * 120)