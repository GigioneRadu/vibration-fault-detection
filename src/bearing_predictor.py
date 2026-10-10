from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from vibration_features import FEATURE_COLUMNS


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "random_forest_multiclass_v3.joblib"
)


# ============================================================
# PREDICTOR CLASS
# ============================================================

class BearingPredictor:

    def __init__(
        self,
        model_path=MODEL_FILE,
        confidence_threshold=0.60,
    ):
        self.model_path = Path(
            model_path
        )

        self.confidence_threshold = (
            confidence_threshold
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found:\n"
                f"{self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )


    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    def _prepare_feature_dataframe(
        self,
        features,
    ):
        """
        Convert a feature dictionary into the exact
        feature order expected by the trained model.
        """

        missing_features = [
            feature_name
            for feature_name in FEATURE_COLUMNS
            if feature_name not in features
        ]

        if missing_features:
            raise ValueError(
                "Missing features: "
                + ", ".join(
                    missing_features
                )
            )

        feature_df = pd.DataFrame(
            [
                {
                    feature_name:
                        features[
                            feature_name
                        ]
                    for feature_name
                    in FEATURE_COLUMNS
                }
            ]
        )

        return feature_df


    # ========================================================
    # CONDITION STATE
    # ========================================================

    def _determine_condition_state(
        self,
        predicted_class,
        confidence,
    ):
        """
        Convert model prediction into an operational state.
        """

        if (
            confidence
            < self.confidence_threshold
        ):
            return "CHECK"

        if predicted_class == "healthy":
            return "NORMAL"

        return "FAULT"


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        features,
    ):
        """
        Predict bearing condition from extracted vibration
        features.

        Returns
        -------
        dict
            predicted_class
            confidence
            condition_state
            probabilities
        """

        feature_df = (
            self._prepare_feature_dataframe(
                features
            )
        )

        predicted_class = (
            self.model.predict(
                feature_df
            )[0]
        )

        probabilities = (
            self.model.predict_proba(
                feature_df
            )[0]
        )

        classes = (
            self.model.classes_
        )

        confidence = float(
            np.max(
                probabilities
            )
        )

        condition_state = (
            self._determine_condition_state(
                predicted_class,
                confidence,
            )
        )

        probability_dictionary = {
            class_name: float(
                probability
            )
            for class_name, probability
            in zip(
                classes,
                probabilities,
            )
        }

        probability_dictionary = dict(
            sorted(
                probability_dictionary.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        return {
            "predicted_class":
                predicted_class,

            "confidence":
                confidence,

            "condition_state":
                condition_state,

            "probabilities":
                probability_dictionary,
        }