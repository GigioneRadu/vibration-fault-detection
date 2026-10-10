from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)


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
# MODEL
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
)


# ---------------------------------------------------------
# CROSS-VALIDATION
# ---------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_accuracy = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)

cv_f1_macro = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
)


# ---------------------------------------------------------
# TRAIN FINAL MODEL
# ---------------------------------------------------------

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
# PRINT CROSS-VALIDATION RESULTS
# ---------------------------------------------------------

print("=" * 90)
print("MULTICLASS MODEL - ADVANCED EVALUATION")
print("=" * 90)

print()
print("5-FOLD CROSS-VALIDATION")
print("-" * 90)

print(
    f"Accuracy mean: "
    f"{cv_accuracy.mean():.4f}"
)

print(
    f"Accuracy std:  "
    f"{cv_accuracy.std():.4f}"
)

print(
    f"F1 macro mean: "
    f"{cv_f1_macro.mean():.4f}"
)

print(
    f"F1 macro std:  "
    f"{cv_f1_macro.std():.4f}"
)


# ---------------------------------------------------------
# TEST REPORT
# ---------------------------------------------------------

print()
print("=" * 90)
print("TEST SET CLASSIFICATION REPORT")
print("=" * 90)

print(
    classification_report(
        y_test,
        y_pred,
    )
)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

labels = [
    "healthy",
    "outer_race",
    "inner_race",
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
)

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
    "Multiclass Model - Confusion Matrix"
)

plt.tight_layout()

cm_path = (
    FIGURES_DIR
    / "multiclass_advanced_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
)

plt.show()


# ---------------------------------------------------------
# PERMUTATION IMPORTANCE
# ---------------------------------------------------------

permutation_result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=20,
    random_state=42,
    scoring="accuracy",
    n_jobs=-1,
)

permutation_df = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance_mean": (
            permutation_result.importances_mean
        ),
        "importance_std": (
            permutation_result.importances_std
        ),
    }
)

permutation_df = (
    permutation_df
    .sort_values(
        by="importance_mean",
        ascending=False,
    )
    .reset_index(drop=True)
)


print()
print("=" * 90)
print("MULTICLASS PERMUTATION IMPORTANCE")
print("=" * 90)

print(
    permutation_df.to_string(
        index=False
    )
)

print("=" * 90)


# ---------------------------------------------------------
# PERMUTATION IMPORTANCE PLOT
# ---------------------------------------------------------

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    permutation_df["feature"],
    permutation_df["importance_mean"],
    xerr=permutation_df["importance_std"],
)

plt.gca().invert_yaxis()

plt.xlabel(
    "Decrease in Accuracy"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Permutation Importance - Multiclass Random Forest"
)

plt.grid(
    axis="x",
)

plt.tight_layout()

permutation_path = (
    FIGURES_DIR
    / "multiclass_permutation_importance.png"
)

plt.savefig(
    permutation_path,
    dpi=300,
)

plt.show()


# ---------------------------------------------------------
# PREDICTION CONFIDENCE
# ---------------------------------------------------------

probabilities = model.predict_proba(
    X_test
)

probability_df = pd.DataFrame(
    probabilities,
    columns=model.classes_,
    index=X_test.index,
)

max_probability = (
    probability_df.max(axis=1)
)

prediction_details = pd.DataFrame(
    {
        "true_label": y_test,
        "predicted_label": y_pred,
        "confidence": max_probability,
    },
    index=X_test.index,
)


# ---------------------------------------------------------
# LOW-CONFIDENCE PREDICTIONS
# ---------------------------------------------------------

low_confidence = prediction_details[
    prediction_details["confidence"]
    < 0.60
]

print()
print("=" * 90)
print("LOW-CONFIDENCE PREDICTIONS (< 0.60)")
print("=" * 90)

if low_confidence.empty:

    print(
        "No low-confidence predictions."
    )

else:

    print(
        low_confidence
        .sort_values(
            by="confidence"
        )
        .to_string()
    )

print("=" * 90)


# ---------------------------------------------------------
# SAVED FILES
# ---------------------------------------------------------

print()
print("Figures saved to:")

print(cm_path)
print(permutation_path)