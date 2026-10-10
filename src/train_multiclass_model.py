from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_bearing_multiclass.csv"
)

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

df = pd.read_csv(DATA_FILE)


# ---------------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------------

feature_columns = [
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

X = df[feature_columns]

y = df["label"]


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ---------------------------------------------------------
# RANDOM FOREST MODEL
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
)


# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------

y_pred = model.predict(
    X_test
)


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred,
)

labels = [
    "healthy",
    "outer_race",
    "inner_race",
]

report = classification_report(
    y_test,
    y_pred,
    labels=labels,
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
)


# ---------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------

print("=" * 90)
print("RANDOM FOREST - MULTICLASS BEARING FAULT CLASSIFICATION")
print("=" * 90)

print()
print(f"Dataset size:   {len(df)} samples")
print(f"Training size:  {len(X_train)} samples")
print(f"Test size:      {len(X_test)} samples")

print()
print(f"Accuracy: {accuracy:.4f}")

print()
print("CLASSIFICATION REPORT")
print("-" * 90)

print(report)

print("=" * 90)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Healthy",
        "Outer Race",
        "Inner Race",
    ],
)

display.plot(
    values_format="d",
)

plt.title(
    "Random Forest - Multiclass Confusion Matrix"
)

plt.tight_layout()

confusion_matrix_path = (
    FIGURES_DIR
    / "multiclass_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
)

plt.show()


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

feature_importance_df = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }
)

feature_importance_df = (
    feature_importance_df
    .sort_values(
        by="importance",
        ascending=False,
    )
    .reset_index(drop=True)
)


print()
print("=" * 90)
print("MULTICLASS FEATURE IMPORTANCE")
print("=" * 90)

print(
    feature_importance_df.to_string(
        index=False
    )
)

print("=" * 90)


# ---------------------------------------------------------
# FEATURE IMPORTANCE PLOT
# ---------------------------------------------------------

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    feature_importance_df["feature"],
    feature_importance_df["importance"],
)

plt.gca().invert_yaxis()

plt.title(
    "Random Forest - Multiclass Feature Importance"
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.grid(
    axis="x",
)

plt.tight_layout()

feature_importance_path = (
    FIGURES_DIR
    / "multiclass_feature_importance.png"
)

plt.savefig(
    feature_importance_path,
    dpi=300,
)

plt.show()


# ---------------------------------------------------------
# PREDICTION PROBABILITIES
# ---------------------------------------------------------

probabilities = model.predict_proba(
    X_test
)

probability_df = pd.DataFrame(
    probabilities,
    columns=model.classes_,
    index=X_test.index,
)

prediction_details = pd.DataFrame(
    {
        "true_label": y_test,
        "predicted_label": y_pred,
    },
    index=X_test.index,
)

prediction_details = pd.concat(
    [
        prediction_details,
        probability_df,
    ],
    axis=1,
)


# ---------------------------------------------------------
# SHOW MISCLASSIFIED SAMPLES
# ---------------------------------------------------------

misclassified = prediction_details[
    prediction_details["true_label"]
    != prediction_details["predicted_label"]
]

print()
print("=" * 90)
print("MISCLASSIFIED SAMPLES")
print("=" * 90)

if misclassified.empty:

    print(
        "No misclassified samples."
    )

else:

    print(
        misclassified.to_string()
    )

print("=" * 90)


# ---------------------------------------------------------
# SAVED FILES
# ---------------------------------------------------------

print()
print("Figures saved to:")

print(confusion_matrix_path)
print(feature_importance_path)