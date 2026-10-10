from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "synthetic_bearing_multiclass_v3.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "random_forest_multiclass_v3.joblib"
)

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK FILES
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

TARGET_COLUMN = "label"

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

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load(MODEL_FILE)


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("5-CLASS MODEL V3 - ADVANCED EVALUATION")
print("=" * 100)

print()
print(f"Dataset size: {len(df)} samples")
print(f"Number of features: {len(FEATURE_COLUMNS)}")
print()

print("Classes:")
for label in sorted(y.unique()):
    print(f"  {label}")

print("=" * 100)


# ============================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================

print()
print("5-FOLD CROSS-VALIDATION")
print("-" * 100)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

accuracy_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy",
)

f1_macro_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="f1_macro",
)

print(f"Accuracy mean: {accuracy_scores.mean():.4f}")
print(f"Accuracy std:  {accuracy_scores.std():.4f}")

print()

print(f"F1 macro mean: {f1_macro_scores.mean():.4f}")
print(f"F1 macro std:  {f1_macro_scores.std():.4f}")

print("=" * 100)


# ============================================================
# TEST SET RECONSTRUCTION
# ============================================================

# Folosim aceeași logică de split ca la antrenare.
# Dacă train_multiclass_model_v3.py folosește alt random_state,
# acesta trebuie modificat aici să fie identic.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred,
)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
)


# ============================================================
# TEST SET REPORT
# ============================================================

print()
print("TEST SET CLASSIFICATION REPORT")
print("=" * 100)

print(
    classification_report(
        y_test,
        y_pred,
        digits=2,
    )
)

print(f"Test accuracy: {accuracy:.4f}")
print(f"Test F1 macro: {f1_macro:.4f}")

print("=" * 100)


# ============================================================
# CONFUSION MATRIX
# ============================================================

class_order = [
    "healthy",
    "outer_race",
    "inner_race",
    "ball_fault",
    "cage_fault",
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_order,
)

display_labels = [
    "Healthy",
    "Outer Race",
    "Inner Race",
    "Ball Fault",
    "Cage Fault",
]

plt.figure(figsize=(9, 7))

plt.imshow(cm)

plt.title(
    "5-Class Model V3 - Confusion Matrix"
)

plt.xlabel(
    "Predicted label"
)

plt.ylabel(
    "True label"
)

plt.xticks(
    np.arange(len(display_labels)),
    display_labels,
    rotation=30,
    ha="right",
)

plt.yticks(
    np.arange(len(display_labels)),
    display_labels,
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
        )

plt.colorbar()

plt.tight_layout()

CONFUSION_MATRIX_FILE = (
    FIGURES_DIR
    / "multiclass_v3_advanced_confusion_matrix.png"
)

plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

print()
print("5-CLASS PERMUTATION IMPORTANCE")
print("=" * 100)

perm_result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=20,
    random_state=42,
    scoring="accuracy",
)

perm_df = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }
)

perm_df = perm_df.sort_values(
    by="importance_mean",
    ascending=False,
)

print(
    perm_df.to_string(
        index=False
    )
)

print("=" * 100)


# ============================================================
# PERMUTATION IMPORTANCE PLOT
# ============================================================

plot_df = perm_df.sort_values(
    by="importance_mean",
    ascending=True,
)

plt.figure(
    figsize=(11, 8)
)

plt.barh(
    plot_df["feature"],
    plot_df["importance_mean"],
    xerr=plot_df["importance_std"],
)

plt.title(
    "Permutation Importance - 5-Class Random Forest V3"
)

plt.xlabel(
    "Decrease in Accuracy"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

PERMUTATION_FILE = (
    FIGURES_DIR
    / "multiclass_v3_permutation_importance.png"
)

plt.savefig(
    PERMUTATION_FILE,
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# PREDICTION CONFIDENCE
# ============================================================

classes = model.classes_

confidence = np.max(
    y_proba,
    axis=1,
)

predicted_labels = classes[
    np.argmax(
        y_proba,
        axis=1,
    )
]


prediction_df = pd.DataFrame(
    {
        "true_label": y_test.to_numpy(),
        "predicted_label": predicted_labels,
        "confidence": confidence,
    },
    index=y_test.index,
)


for class_index, class_name in enumerate(classes):
    prediction_df[class_name] = y_proba[:, class_index]


# ============================================================
# LOW-CONFIDENCE PREDICTIONS
# ============================================================

LOW_CONFIDENCE_THRESHOLD = 0.60

low_confidence = prediction_df[
    prediction_df["confidence"]
    < LOW_CONFIDENCE_THRESHOLD
]

print()
print(
    f"LOW-CONFIDENCE PREDICTIONS "
    f"(< {LOW_CONFIDENCE_THRESHOLD:.2f})"
)

print("=" * 100)

if len(low_confidence) == 0:
    print("No low-confidence predictions.")
else:
    print(
        low_confidence
        .sort_values(
            by="confidence"
        )
        .to_string()
    )

print("=" * 100)


# ============================================================
# MISCLASSIFIED SAMPLES
# ============================================================

misclassified = prediction_df[
    prediction_df["true_label"]
    != prediction_df["predicted_label"]
]

print()
print("MISCLASSIFIED SAMPLES")
print("=" * 100)

if len(misclassified) == 0:
    print("No misclassified samples.")
else:
    print(
        misclassified
        .sort_values(
            by="confidence"
        )
        .to_string()
    )

print("=" * 100)


# ============================================================
# CAGE FAULT ANALYSIS
# ============================================================

cage_fault_samples = prediction_df[
    prediction_df["true_label"]
    == "cage_fault"
]

correct_cage_fault = cage_fault_samples[
    cage_fault_samples["predicted_label"]
    == "cage_fault"
]

incorrect_cage_fault = cage_fault_samples[
    cage_fault_samples["predicted_label"]
    != "cage_fault"
]

total_cage_fault = len(
    cage_fault_samples
)

correct_cage_fault_count = len(
    correct_cage_fault
)

incorrect_cage_fault_count = len(
    incorrect_cage_fault
)

if total_cage_fault > 0:
    cage_fault_recall = (
        correct_cage_fault_count
        / total_cage_fault
    )
else:
    cage_fault_recall = np.nan


print()
print("CAGE FAULT DETAILED ANALYSIS")
print("=" * 100)

print(
    f"Total cage fault samples:      "
    f"{total_cage_fault}"
)

print(
    f"Correct cage fault samples:    "
    f"{correct_cage_fault_count}"
)

print(
    f"Incorrect cage fault samples:  "
    f"{incorrect_cage_fault_count}"
)

print(
    f"Cage fault recall:             "
    f"{cage_fault_recall:.4f}"
)

print()

if incorrect_cage_fault_count > 0:

    print(
        "Cage fault misclassification destinations:"
    )

    print(
        incorrect_cage_fault[
            "predicted_label"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Misclassified cage fault samples:"
    )

    print(
        incorrect_cage_fault
        .sort_values(
            by="confidence"
        )
        .to_string()
    )

else:
    print(
        "All cage fault samples "
        "were classified correctly."
    )

print("=" * 100)


# ============================================================
# SUMMARY
# ============================================================

print()
print("MODEL V3 EVALUATION SUMMARY")
print("=" * 100)

print(
    f"Cross-validation accuracy: "
    f"{accuracy_scores.mean():.4f} "
    f"+/- {accuracy_scores.std():.4f}"
)

print(
    f"Cross-validation F1 macro: "
    f"{f1_macro_scores.mean():.4f} "
    f"+/- {f1_macro_scores.std():.4f}"
)

print(
    f"Test accuracy:              "
    f"{accuracy:.4f}"
)

print(
    f"Test F1 macro:              "
    f"{f1_macro:.4f}"
)

print(
    f"Cage fault recall:          "
    f"{cage_fault_recall:.4f}"
)

print("=" * 100)


# ============================================================
# SAVED FILES
# ============================================================

print()
print("Figures saved to:")

print(
    CONFUSION_MATRIX_FILE
)

print(
    PERMUTATION_FILE
)