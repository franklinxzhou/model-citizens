import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

# f2_preprocessor
# Speciality preprocessor for TabM model

num_cols = [
    "year_of_born",
    "email_or_tel_available",
    "safety_rating",
    "annual_income",
    "high_education_ind",
    "address_change_ind",
    "past_num_of_claims",
    "liab_prct",
    "policy_report_filed_ind",
    "claim_est_payout",
    "vehicle_made_year",
    "vehicle_price",
    "vehicle_weight",
    "age_of_DL",
    "vehicle_mileage",
]

cat_cols = [
    "gender",
    "living_status",
    "zip_code",
    "claim_day_of_week",
    "accident_site",
    "witness_present_ind",
    "channel",
    "vehicle_category",
    "vehicle_color",
    "accident_type",
    "in_network_bodyshop",
]

class Preprocessor:
    """
    Modular preprocessing pipeline for TabM:

    ✔ Domain features
    ✔ Numeric features → median impute
    ✔ Categorical → vocab → integer codes
    ✔ Target: reshape to (N,1)
    """

    def __init__(
        self,
        num_cols: List[str],
        cat_cols: List[str],
        target_col: str = "subrogation",
    ):
        # Base (raw) numeric columns
        self.base_num_cols = num_cols

        # Categorical columns
        self.cat_cols = cat_cols

        # Target
        self.target_col = target_col

        # Domain numeric features we create
        self.domain_num_cols = [
            "driver_age_at_claim",
            "vehicle_age_at_claim",
            "years_with_license",
            "log_claim_est_payout",
        ]

        # Final numeric columns (base + domain)
        self.num_cols: List[str] = []

        # Learned during fit()
        self.num_medians: Dict[str, float] = {}
        self.cat_categories_: Dict[str, List[str]] = {}
        self.cat_cardinalities_: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # 1) DOMAIN FEATURE ENGINEERING
    # ------------------------------------------------------------------ #
    def _add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Parse claim_date to year
        claim_dt = pd.to_datetime(df.get("claim_date"), errors="coerce")
        claim_year = claim_dt.dt.year

        # 1. driver_age_at_claim
        df["driver_age_at_claim"] = claim_year - df["year_of_born"]
        df.loc[df["driver_age_at_claim"] < 0, "driver_age_at_claim"] = np.nan

        # 2. vehicle_age_at_claim
        if "age_of_vehicle" in df.columns:
            df["vehicle_age_at_claim"] = df["age_of_vehicle"]
        else:
            df["vehicle_age_at_claim"] = claim_year - df["vehicle_made_year"]
        df.loc[df["vehicle_age_at_claim"] < 0, "vehicle_age_at_claim"] = np.nan

        # 3. years_with_license
        df["years_with_license"] = (
            df["driver_age_at_claim"] - df["age_of_DL"]
        )
        df.loc[df["years_with_license"] < 0, "years_with_license"] = np.nan

        # 4. log_claim_est_payout
        payout = df["claim_est_payout"].clip(lower=0)
        df["log_claim_est_payout"] = np.log1p(payout)

        return df

    # ------------------------------------------------------------------ #
    # 2) NUMERIC PREPROCESSING
    # ------------------------------------------------------------------ #
    def _process_numeric(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply median imputation and return numeric matrix (float32).
        """
        num_df = df[self.num_cols].copy()

        for col, med in self.num_medians.items():
            if col in num_df.columns:
                num_df[col] = num_df[col].fillna(med)

        return num_df.to_numpy(dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 3) CATEGORICAL PREPROCESSING
    # ------------------------------------------------------------------ #
    def _process_categorical(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert categorical strings into integer codes (0..C-1).
        """
        X_cat_list = []

        for col in self.cat_cols:
            series = df[col].astype("object")
            series = series.where(~series.isna(), "MISSING")

            cats = self.cat_categories_[col]
            cat_to_idx = {cat: i for i, cat in enumerate(cats)}

            codes = series.map(
                lambda x: cat_to_idx.get(x, cat_to_idx["MISSING"])
            ).to_numpy(dtype=np.int64)

            X_cat_list.append(codes)

        if X_cat_list:
            return np.stack(X_cat_list, axis=1)
        else:
            return np.empty((len(df), 0), dtype=np.int64)

    # ------------------------------------------------------------------ #
    # 4) FIT: learn medians + categorical vocab
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame):
        df = self._add_domain_features(df)

        # numeric columns = base + domain
        all_num = list(dict.fromkeys(self.base_num_cols + self.domain_num_cols))
        self.num_cols = [c for c in all_num if c in df.columns]

        # numeric medians
        num_df = df[self.num_cols]
        self.num_medians = num_df.median(numeric_only=True).to_dict()

        # categorical vocab
        for col in self.cat_cols:
            series = df[col].astype("object")
            series = series.where(~series.isna(), "MISSING")

            cats = pd.Categorical(series).categories.tolist()
            if "MISSING" not in cats:
                cats.append("MISSING")

            self.cat_categories_[col] = cats
            self.cat_cardinalities_[col] = len(cats)

        return self

    # ------------------------------------------------------------------ #
    # 5) TRANSFORM: full numeric + categorical preprocessing
    # ------------------------------------------------------------------ #
    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:

        df = self._add_domain_features(df)

        # numeric
        X_num = self._process_numeric(df)

        # categorical
        X_cat = self._process_categorical(df)

        # target
        y = None
        if self.target_col in df.columns:
            y = (
                df[self.target_col]
                .to_numpy(dtype=np.float32)
                .reshape(-1, 1)
            )

        return X_num, X_cat, y

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        self.fit(df)
        return self.transform(df)

