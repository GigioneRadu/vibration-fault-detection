from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "random_forest_multiclass_v3.joblib"
)

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_bearing_multiclass_v3.csv"
)


# ============================================================
# FEATURE ORDER
# ============================================================

FEATURE_COLUMNS = [
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "shaft_amplitude",
    "bpfo_raw",
    "bpfi_raw",
    "bsf_raw",
    "ftf_raw",
    "bpfo_envelope",
    "bpfi_envelope",
    "bsf_envelope",
    "ftf_envelope",
]


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_FILE}"
    )

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_FILE)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_FILE)


# ============================================================
# SELECT ONE SAMPLE
# ============================================================

# Pentru primul test luăm un exemplu din dataset.
# Mai târziu această parte va fi înlocuită cu features
# extrase dintr-un semnal real de vibrație.

sample_index = 0

sample = df.loc[
    sample_index,
    FEATURE_COLUMNS,
]


# ============================================================
# PREPARE SAMPLE FOR MODEL
# ============================================================

sample_df = pd.DataFrame(
    [sample],
    columns=FEATURE_COLUMNS,
)


# ============================================================
# MODEL PREDICTION
# ============================================================

predicted_class = model.predict(
    sample_df
)[0]

probabilities = model.predict_proba(
    sample_df
)[0]


# ============================================================
# PROBABILITY TABLE
# ============================================================

probability_table = pd.DataFrame(
    {
        "class": model.classes_,
        "probability": probabilities,
    }
)

probability_table = (
    probability_table
    .sort_values(
        by="probability",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# CONFIDENCE
# ============================================================

confidence = probability_table.loc[
    0,
    "probability",
]


# ============================================================
# CONDITION STATE
# ============================================================

# Prima versiune a logicii operaționale.
#
# healthy + probabilitate mare -> NORMAL
# probabilitate mică -> CHECK
# defect + probabilitate mare -> FAULT

if confidence < 0.60:
    condition_state = "CHECK"

elif predicted_class == "healthy":
    condition_state = "NORMAL"

else:
    condition_state = "FAULT"


# ============================================================
# TRUE LABEL
# ============================================================

true_label = df.loc[
    sample_index,
    "label",
]


# ============================================================
# OUTPUT
# ============================================================

print("=" * 100)
print("BEARING CONDITION PREDICTION")
print("=" * 100)

print()
print(f"Sample index:    {sample_index}")
print(f"True label:      {true_label}")
print(f"Predicted class: {predicted_class}")
print(f"Confidence:      {confidence:.4f}")
print(f"Condition state: {condition_state}")

print()
print("CLASS PROBABILITIES")
print("-" * 100)

for _, row in probability_table.iterrows():

    print(
        f"{row['class']:<15} "
        f"{row['probability']:.4f}"
    )

print("=" * 100)


# ============================================================
# FEATURE VALUES
# ============================================================

print()
print("INPUT FEATURES")
print("=" * 100)

for feature_name in FEATURE_COLUMNS:

    value = sample[
        feature_name
    ]

    print(
        f"{feature_name:<20} "
        f"{value:.6f}"
    )

print("=" * 100)