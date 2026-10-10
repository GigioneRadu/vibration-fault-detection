from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
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
    / "synthetic_bearing_dataset_v2.csv"
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

# Convert labels to binary values:
# healthy = 0
# outer_race = 1
y = (
    df["label"]
    .map(
        {
            "healthy": 0,
            "outer_race": 1,
        }
    )
)


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
    n_estimators=300,
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

cv_recall = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="recall",
    n_jobs=-1,
)

cv_roc_auc = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
)


# ---------------------------------------------------------
# TRAIN FINAL MODEL ON TRAINING SET
# ---------------------------------------------------------

model.fit(
    X_train,
    y_train,
)


# ---------------------------------------------------------
# PROBABILITIES
# ---------------------------------------------------------

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ---------------------------------------------------------
# DEFAULT THRESHOLD
# ---------------------------------------------------------

default_threshold = 0.50

pred_default = (
    probabilities
    >= default_threshold
).astype(int)


# ---------------------------------------------------------
# LOWER THRESHOLD
# ---------------------------------------------------------

sensitive_threshold = 0.35

pred_sensitive = (
    probabilities
    >= sensitive_threshold
).astype(int)


# ---------------------------------------------------------
# PRINT CROSS-VALIDATION RESULTS
# ---------------------------------------------------------

print("=" * 85)
print("MODEL V2 - ADVANCED EVALUATION")
print("=" * 85)

print()
print("5-FOLD CROSS-VALIDATION")
print("-" * 85)

print(
    f"Accuracy mean: "
    f"{cv_accuracy.mean():.4f}"
)

print(
    f"Accuracy std:  "
    f"{cv_accuracy.std():.4f}"
)

print(
    f"Recall mean:   "
    f"{cv_recall.mean():.4f}"
)

print(
    f"Recall std:    "
    f"{cv_recall.std():.4f}"
)

print(
    f"ROC-AUC mean:  "
    f"{cv_roc_auc.mean():.4f}"
)

print(
    f"ROC-AUC std:   "
    f"{cv_roc_auc.std():.4f}"
)


# ---------------------------------------------------------
# ROC-AUC ON TEST SET
# ---------------------------------------------------------

test_roc_auc = roc_auc_score(
    y_test,
    probabilities,
)

print()
print(
    f"Test ROC-AUC:  "
    f"{test_roc_auc:.4f}"
)


# ---------------------------------------------------------
# DEFAULT THRESHOLD REPORT
# ---------------------------------------------------------

print()
print("=" * 85)
print(
    f"DEFAULT THRESHOLD = "
    f"{default_threshold:.2f}"
)
print("=" * 85)

print(
    classification_report(
        y_test,
        pred_default,
        target_names=[
            "Healthy",
            "Outer Race",
        ],
    )
)


# ---------------------------------------------------------
# SENSITIVE THRESHOLD REPORT
# ---------------------------------------------------------

print()
print("=" * 85)
print(
    f"SENSITIVE THRESHOLD = "
    f"{sensitive_threshold:.2f}"
)
print("=" * 85)

print(
    classification_report(
        y_test,
        pred_sensitive,
        target_names=[
            "Healthy",
            "Outer Race",
        ],
    )
)


# ---------------------------------------------------------
# CONFUSION MATRICES
# ---------------------------------------------------------

cm_default = confusion_matrix(
    y_test,
    pred_default,
)

cm_sensitive = confusion_matrix(
    y_test,
    pred_sensitive,
)


print()
print("Confusion matrix - threshold 0.50")
print(cm_default)

print()
print("Confusion matrix - threshold 0.35")
print(cm_sensitive)


# ---------------------------------------------------------
# ROC CURVE
# ---------------------------------------------------------

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    probabilities,
)

roc_auc_value = auc(
    fpr,
    tpr,
)

plt.figure(
    figsize=(9, 7)
)

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=(
        f"Random Forest "
        f"(AUC = {roc_auc_value:.3f})"
    ),
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Random classifier",
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Random Forest V2"
)

plt.grid(True)
plt.legend()

plt.tight_layout()

roc_path = (
    FIGURES_DIR
    / "v2_roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300,
)

plt.show()


# ---------------------------------------------------------
# PRECISION-RECALL CURVE
# ---------------------------------------------------------

precision, recall, pr_thresholds = (
    precision_recall_curve(
        y_test,
        probabilities,
    )
)

plt.figure(
    figsize=(9, 7)
)

plt.plot(
    recall,
    precision,
    linewidth=2,
)

plt.xlabel(
    "Recall - Outer Race Fault"
)

plt.ylabel(
    "Precision - Outer Race Fault"
)

plt.title(
    "Precision-Recall Curve - Random Forest V2"
)

plt.grid(True)

plt.tight_layout()

pr_path = (
    FIGURES_DIR
    / "v2_precision_recall_curve.png"
)

plt.savefig(
    pr_path,
    dpi=300,
)

plt.show()


# ---------------------------------------------------------
# CONFUSION MATRIX - DEFAULT THRESHOLD
# ---------------------------------------------------------

display_default = ConfusionMatrixDisplay(
    confusion_matrix=cm_default,
    display_labels=[
        "Healthy",
        "Outer Race",
    ],
)

display_default.plot(
    values_format="d",
)

plt.title(
    "Confusion Matrix - Threshold 0.50"
)

plt.tight_layout()

cm_default_path = (
    FIGURES_DIR
    / "v2_confusion_matrix_threshold_050.png"
)

plt.savefig(
    cm_default_path,
    dpi=300,
)

plt.show()


# ---------------------------------------------------------
# CONFUSION MATRIX - SENSITIVE THRESHOLD
# ---------------------------------------------------------

display_sensitive = ConfusionMatrixDisplay(
    confusion_matrix=cm_sensitive,
    display_labels=[
        "Healthy",
        "Outer Race",
    ],
)

display_sensitive.plot(
    values_format="d",
)

plt.title(
    "Confusion Matrix - Threshold 0.35"
)

plt.tight_layout()

cm_sensitive_path = (
    FIGURES_DIR
    / "v2_confusion_matrix_threshold_035.png"
)

plt.savefig(
    cm_sensitive_path,
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
print("=" * 85)
print("PERMUTATION IMPORTANCE")
print("=" * 85)

print(
    permutation_df.to_string(
        index=False
    )
)

print("=" * 85)


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
    "Permutation Importance - Random Forest V2"
)

plt.grid(
    axis="x",
)

plt.tight_layout()

permutation_path = (
    FIGURES_DIR
    / "v2_permutation_importance.png"
)

plt.savefig(
    permutation_path,
    dpi=300,
)

plt.show()


# ---------------------------------------------------------
# SAVED FIGURES
# ---------------------------------------------------------

print()
print("Figures saved to:")

print(roc_path)
print(pr_path)
print(cm_default_path)
print(cm_sensitive_path)
print(permutation_path)