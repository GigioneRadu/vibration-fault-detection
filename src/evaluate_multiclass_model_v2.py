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
    / "synthetic_bearing_multiclass_v2.csv"
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
    n_estimators=500,
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
# TEST PREDICTIONS
# ---------------------------------------------------------

y_pred = model.predict(
    X_test
)


# ---------------------------------------------------------
# LABEL ORDER
# ---------------------------------------------------------

labels = [
    "healthy",
    "outer_race",
    "inner_race",
    "ball_fault",
]


# ---------------------------------------------------------
# PRINT CROSS-VALIDATION RESULTS
# ---------------------------------------------------------

print("=" * 95)
print("4-CLASS MODEL - ADVANCED EVALUATION")
print("=" * 95)

print()
print("5-FOLD CROSS-VALIDATION")
print("-" * 95)

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
print("=" * 95)
print("TEST SET CLASSIFICATION REPORT")
print("=" * 95)

print(
    classification_report(
        y_test,
        y_pred,
        labels=labels,
    )
)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

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
        "Ball Fault",
    ],
)

display.plot(
    values_format="d",
)

plt.title(
    "4-Class Model - Confusion Matrix"
)

plt.tight_layout()

cm_path = (
    FIGURES_DIR
    / "multiclass_v2_advanced_confusion_matrix.png"
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
print("=" * 95)
print("4-CLASS PERMUTATION IMPORTANCE")
print("=" * 95)

print(
    permutation_df.to_string(
        index=False
    )
)

print("=" * 95)


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
    "Permutation Importance - 4-Class Random Forest"
)

plt.grid(
    axis="x",
)

plt.tight_layout()

permutation_path = (
    FIGURES_DIR
    / "multiclass_v2_permutation_importance.png"
)

plt.savefig(
    permutation_path,
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

max_probability = (
    probability_df.max(axis=1)
)


# ---------------------------------------------------------
# PREDICTION DETAILS
# ---------------------------------------------------------

prediction_details = pd.DataFrame(
    {
        "true_label": y_test,
        "predicted_label": y_pred,
        "confidence": max_probability,
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
# LOW-CONFIDENCE PREDICTIONS
# ---------------------------------------------------------

low_confidence = prediction_details[
    prediction_details["confidence"]
    < 0.60
]

print()
print("=" * 95)
print("LOW-CONFIDENCE PREDICTIONS (< 0.60)")
print("=" * 95)

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

print("=" * 95)


# ---------------------------------------------------------
# MISCLASSIFIED SAMPLES
# ---------------------------------------------------------

misclassified = prediction_details[
    prediction_details["true_label"]
    != prediction_details["predicted_label"]
]

print()
print("=" * 95)
print("MISCLASSIFIED SAMPLES")
print("=" * 95)

if misclassified.empty:

    print(
        "No misclassified samples."
    )

else:

    print(
        misclassified
        .sort_values(
            by="confidence"
        )
        .to_string()
    )

print("=" * 95)


# ---------------------------------------------------------
# SAVED FILES
# ---------------------------------------------------------

print()
print("Figures saved to:")

print(cm_path)
print(permutation_path)