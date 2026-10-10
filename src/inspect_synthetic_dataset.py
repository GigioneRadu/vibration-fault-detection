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
    / "synthetic_bearing_dataset.csv"
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

print("=" * 80)
print("SYNTHETIC DATASET INSPECTION")
print("=" * 80)

print()
print("Shape:")
print(df.shape)

print()
print("Columns:")
print(df.columns.tolist())

print()
print("Class distribution:")
print(df["label"].value_counts())

print()
print("Missing values:")
print(df.isna().sum())

print()
print("First rows:")
print(df.head())

print()
print("Descriptive statistics:")
print(df.describe(include="all"))

print("=" * 80)


# ---------------------------------------------------------
# FEATURES TO INSPECT
# ---------------------------------------------------------

features_to_plot = [
    "rms",
    "crest_factor",
    "kurtosis",
    "bpfo_raw",
    "bpfo_envelope",
]


# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------

for feature in features_to_plot:

    plt.figure(figsize=(10, 6))

    for label in ["healthy", "outer_race"]:

        subset = df[
            df["label"] == label
        ][feature]

        plt.hist(
            subset,
            bins=30,
            alpha=0.6,
            label=label,
        )

    plt.title(
        f"Feature Distribution: {feature}"
    )

    plt.xlabel(feature)
    plt.ylabel("Count")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    figure_path = (
        FIGURES_DIR
        / f"distribution_{feature}.png"
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
print("=" * 80)
print("MEAN FEATURE VALUES BY CLASS")
print("=" * 80)

print(grouped_means)

print("=" * 80)