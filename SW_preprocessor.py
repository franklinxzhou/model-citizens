"""
robust_preprocessor.py

Feature engineering + robust preprocessing utilities for the TriGuard project.

This module provides:
- enhanced_feature_engineer: light domain feature engineering for claims data
- RobustPreprocessor: label-encoding + column alignment for tabular models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEASON_MAP = {
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Fall",
    10: "Fall",
    11: "Fall",
    12: "Winter",
}

LIABILITY_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
LIABILITY_LABELS = [
    "0-10",
    "10-20",
    "20-30",
    "30-40",
    "40-50",
    "50-60",
    "60-70",
    "70-80",
    "80-90",
    "90-100",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def enhanced_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply light domain feature engineering to the raw claims DataFrame.

    This function is model-agnostic and safe to use before tree models,
    TabM, etc. It creates temporal, interaction, and log features and
    normalizes some categorical types.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame with original columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered features.
    """
    df_fe = df.copy()

    # ------------------------------------------------------------------
    # Date / temporal features
    # ------------------------------------------------------------------
    df_fe["claim_date"] = pd.to_datetime(df_fe["claim_date"], errors="coerce")
    df_fe["claim_year"] = df_fe["claim_date"].dt.year
    df_fe["claim_month"] = df_fe["claim_date"].dt.month
    df_fe["claim_day"] = df_fe["claim_date"].dt.day
    df_fe["claim_dayofweek"] = df_fe["claim_date"].dt.dayofweek
    df_fe["claim_quarter"] = df_fe["claim_date"].dt.quarter
    df_fe["is_weekend"] = (df_fe["claim_dayofweek"] >= 5).astype(int)

    # Season
    df_fe["claim_month"] = df_fe["claim_month"].fillna(1)
    df_fe["season"] = df_fe["claim_month"].map(SEASON_MAP)

    # ------------------------------------------------------------------
    # Binary flags
    # ------------------------------------------------------------------
    df_fe["witness_binary"] = (df_fe["witness_present_ind"] == "Y").astype(int)
    # policy_report_filed_ind is already 0/1 in the original data
    df_fe["police_binary"] = df_fe["policy_report_filed_ind"]

    df_fe["multicar_binary"] = df_fe["accident_type"].isin(
        ["multi_vehicle_clear", "multi_vehicle_unclear"]
    ).astype(int)

    # ------------------------------------------------------------------
    # Interaction / score features
    # ------------------------------------------------------------------
    df_fe["liab_x_multicar"] = df_fe["liab_prct"] * df_fe["multicar_binary"]
    df_fe["evidence_score"] = df_fe["witness_binary"] + df_fe["police_binary"]

    # Liability buckets (categorical)
    df_fe["liability_bucket"] = pd.cut(
        df_fe["liab_prct"],
        bins=LIABILITY_BINS,
        labels=LIABILITY_LABELS,
    )

    # ------------------------------------------------------------------
    # Driver age
    # ------------------------------------------------------------------
    df_fe["driver_age"] = df_fe["claim_year"] - df_fe["year_of_born"]
    df_fe["driver_age"] = df_fe["driver_age"].clip(16, 80)

    # ------------------------------------------------------------------
    # Log transforms
    # ------------------------------------------------------------------
    df_fe["log_payout"] = np.log1p(df_fe["claim_est_payout"])
    df_fe["log_income"] = np.log1p(df_fe["annual_income"])

    # ------------------------------------------------------------------
    # Clean-up
    # ------------------------------------------------------------------
    # Drop original claim_date (we now use derived features)
    df_fe = df_fe.drop(columns=["claim_date"], errors="ignore")

    # Ensure pandas "category" dtypes become strings,
    # making downstream encoders simpler & more robust.
    for col in df_fe.select_dtypes(include=["category"]).columns:
        df_fe[col] = df_fe[col].astype(str)

    return df_fe


# ---------------------------------------------------------------------------
# RobustPreprocessor
# ---------------------------------------------------------------------------

@dataclass
class RobustPreprocessor:
    """
    A robust, label-encoder-based preprocessor for tabular models.

    Responsibilities:
    - Run `enhanced_feature_engineer` to create additional features.
    - Identify all object-type columns as categoricals.
    - Fit a separate LabelEncoder per categorical column, with an explicit
      "__UNK__" category for unseen levels.
    - On transform:
        * Encode categoricals.
        * Align columns to the training-time feature set:
          - Add missing columns with value 0.
          - Drop extra columns.

    This is particularly useful when you need train/test column alignment
    across slightly different versions of the data.
    """

    label_encoders: Dict[str, LabelEncoder] = field(default_factory=dict)
    categorical_cols: List[str] = field(default_factory=list)
    trained_features: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "RobustPreprocessor":
        """
        Fit the preprocessor on a training DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw training DataFrame.

        Returns
        -------
        RobustPreprocessor
            Fitted instance.
        """
        df_fe = enhanced_feature_engineer(df.copy())

        # Identify categorical columns (object dtype)
        self.categorical_cols = df_fe.select_dtypes(include=["object"]).columns.tolist()

        # Fit LabelEncoder per categorical column, with explicit "__UNK__" bucket
        self.label_encoders = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            vals = df_fe[col].astype(str).fillna("Missing").unique().tolist()
            if "__UNK__" not in vals:
                vals.append("__UNK__")
            le.fit(vals)
            self.label_encoders[col] = le

        # Transform once to lock feature order
        df_processed = self._transform_internal(df_fe, is_training=True)
        self.trained_features = df_processed.columns.tolist()

        return self

    # ------------------------------------------------------------------
    # Core transform logic
    # ------------------------------------------------------------------
    def _transform_internal(self, df_fe: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Internal helper to encode categoricals and (optionally) align columns.

        Parameters
        ----------
        df_fe : pd.DataFrame
            Feature-engineered DataFrame.
        is_training : bool, default=False
            If True, do not align to trained_features (used during fit).

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame.
        """
        df_tr = df_fe.copy()

        # Encode categorical vars with their fitted LabelEncoders
        for col, le in self.label_encoders.items():
            if col not in df_tr.columns:
                # If the column is missing entirely, create it as "Missing"
                df_tr[col] = "Missing"

            vals = df_tr[col].astype(str).fillna("Missing")
            known = set(le.classes_)
            # Map unknown categories to "__UNK__"
            vals[~vals.isin(known)] = "__UNK__"
            df_tr[col] = le.transform(vals)

        # At inference time, align columns to training feature set
        if not is_training and self.trained_features:
            # Add any missing training columns with 0
            for c in self.trained_features:
                if c not in df_tr.columns:
                    df_tr[c] = 0

            # Drop extra columns and re-order
            df_tr = df_tr[self.trained_features]

        return df_tr

    # ------------------------------------------------------------------
    # Public transform
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a new DataFrame using the fitted encoders and
        training-time feature order.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input DataFrame.

        Returns
        -------
        pd.DataFrame
            Encoded and column-aligned DataFrame.
        """
        df_fe = enhanced_feature_engineer(df.copy())
        return self._transform_internal(df_fe, is_training=False)
