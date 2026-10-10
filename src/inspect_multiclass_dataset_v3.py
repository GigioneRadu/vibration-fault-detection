from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_bearing_multiclass_v3.csv"
)

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

df = pd.read_csv(DATA_FILE)


# ---------------------------------------------------------
# BASIC INFORMATION
# ---------------------------------------------------------

print("=" * 100)
print("MULTICLASS SYNTHETIC DATASET V3 INSPECTION")
print("=" * 100)

print()
print("Shape:")
print(df.shape)

print()
print("Class distribution:")
print(df["label"].value_counts())

print()
print("Missing values:")
print(df.isna().sum())

print()
print("First rows:")
print(df.head())

print("=" * 100)


# ---------------------------------------------------------
# FEATURES TO INSPECT
# ---------------------------------------------------------

features_to_plot = [
    "rms",
    "crest_factor",
    "kurtosis",
    "bpfo_raw",
    "bpfi_raw",
    "bsf_raw",
    "ftf_raw",
    "bpfo_envelope",
    "bpfi_envelope",
    "bsf_envelope",
    "ftf_envelope",
]

classes = [
    "healthy",
    "outer_race",
    "inner_race",
    "ball_fault",
    "cage_fault",
]


# ---------------------------------------------------------
# DISTRIBUTION PLOTS
# ---------------------------------------------------------

for feature in features_to_plot:

    plt.figure(
        figsize=(10, 6)
    )

    for label in classes:

        subset = df[
            df["label"] == label
        ][feature]

        plt.hist(
            subset,
            bins=30,
            alpha=0.40,
            label=label,
        )

    plt.title(
        f"Multiclass V3 Feature Distribution: {feature}"
    )

    plt.xlabel(feature)
    plt.ylabel("Count")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    figure_path = (
        FIGURES_DIR
        / f"multiclass_v3_distribution_{feature}.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
    )

    plt.show()


# ---------------------------------------------------------
# GROUPED MEANS
# ---------------------------------------------------------

numeric_columns = df.select_dtypes(
    include="number"
).columns

grouped_means = (
    df.groupby("label")[numeric_columns]
    .mean()
    .T
)

print()
print("=" * 100)
print("MEAN FEATURE VALUES BY CLASS")
print("=" * 100)

print(grouped_means)

print("=" * 100)


# ---------------------------------------------------------
# GROUPED MEDIANS
# ---------------------------------------------------------

grouped_medians = (
    df.groupby("label")[numeric_columns]
    .median()
    .T
)

print()
print("=" * 100)
print("MEDIAN FEATURE VALUES BY CLASS")
print("=" * 100)

print(grouped_medians)

print("=" * 100)


# ---------------------------------------------------------
# SELECTED DIAGNOSTIC SUMMARY
# ---------------------------------------------------------

selected_features = [
    "bpfo_envelope",
    "bpfi_envelope",
    "bsf_envelope",
    "ftf_envelope",
]

diagnostic_summary = (
    df.groupby("label")[selected_features]
    .mean()
)

print()
print("=" * 100)
print("DIAGNOSTIC ENVELOPE SUMMARY")
print("=" * 100)

print(diagnostic_summary)

print("=" * 100)


# ---------------------------------------------------------
# SAVED FIGURES
# ---------------------------------------------------------

print()
print("Saved figures:")

for feature in features_to_plot:

    print(
        FIGURES_DIR
        / f"multiclass_v3_distribution_{feature}.png"
    )