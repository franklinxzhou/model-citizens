import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# f1-Preprocessor for CatBoost Model Only
# Unleash the full potential of CatBoost
# Developed based on cc5-Preprocessor

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self): 
        # This list is now just for *identification*
        self.cat_features_ = [
            'accident_site', 'accident_type', 'channel',
            'vehicle_category', 'vehicle_color', 'living_status',
            'claim_day_of_week', 'gender', 'in_network_bodyshop',
            'season', 'policy_report_filed_ind',
            'witness_present_ind'
        ]
        # All other attributes (smoothing_factor, target_mean_maps_, 
        # global_mean_, stats_params_, cols_for_zscore_) are REMOVED.

    def fit(self, X, y=None):
        """
        Fit method. For CatBoost, this preprocessor is 'stateless' 
        as it learns no parameters (no target encoding, no z-scoring).
        """
        print("Fitting CatBoost Preprocessor (stateless)...")
        # No learning is required.
        print("Fit complete.")
        return self
    
    def transform(self, X):
        """
        Applies a *minimalist* set of feature engineering steps 
        and the one mandatory dtype conversion for CatBoost.
        """
        print("Transforming data for CatBoost...")
        df_fe = X.copy()

        # --- 1. Keep Minimalist, Raw Feature Creation ---
        # (These are all the "Keep" or "Modify" functions we identified)
        df_fe = self._clean_data(df_fe)
        df_fe = self._engineer_temporal(df_fe)
        df_fe = self._engineer_accident_evidence(df_fe) # (Modified to remove interactions)
        df_fe = self._engineer_accident_site(df_fe)
        df_fe = self._engineer_driver_age(df_fe)       # (Modified to remove interactions)
        df_fe = self._engineer_vehicle_features(df_fe) # (Modified: *only* 'is_medium_vehicle' etc., no price/weight bins)
        df_fe = self._engineer_ratios(df_fe)           # (Modified to remove binning)
        df_fe = self._engineer_policyholder(df_fe)     # (Modified to remove binning)
        # ... (and any other *raw* feature creation) ...
        
        
        # --- 2. Remove Redundant/Harmful Steps ---
        # (Confirm you are NOT calling any of these)
        # DO NOT CALL: _engineer_liability_buckets (Redundant binning)
        # DO NOT CALL: _engineer_claim_characteristics (Redundant binning/interactions)
        # DO NOT CALL: _engineer_channel (Redundant OHE)
        # DO NOT CALL: _engineer_composite_scores (Redundant interactions)
        # DO NOT CALL: _engineer_statistical_features (Redundant scaling)
        # DO NOT CALL: _engineer_domain_flags (Redundant interactions)
        # DO NOT CALL: _engineer_new_domain_interactions (Redundant interactions)
        # DO NOT CALL: _engineer_new_interactions (Harmful target encoding)
        
        
        # --- 3. ADD: The ONLY Mandatory Transformation Step ---
        print(f"Applying mandatory .astype(str) to: {self.cat_features_}")
        for col in self.cat_features_:
            if col in df_fe.columns:
                # This converts np.nan (a float) to "nan" (a string)
                # This is the CRITICAL fix[cite: 148, 150].
                df_fe[col] = df_fe[col].astype(str)
            else:
                print(f"Warning: Categorical feature '{col}' not found in DataFrame.")

        # --- 4. Final Cleanup ---
        # Drop any remaining datetime objects used for engineering
        datetime_cols_to_drop = df_fe.select_dtypes(include=['datetime64[ns]']).columns
        if len(datetime_cols_to_drop) > 0:
            print(f"Dropping helper datetime columns: {list(datetime_cols_to_drop)}")
            df_fe = df_fe.drop(columns=datetime_cols_to_drop)
            
        print("Transform complete.")
        return df_fe
    
    # Helper functions
    # Only helper functions creating new, raw features are retained
    # All interaction, scaling functions have been dropped

    def _clean_data(self, df):
        """Applies data cleaning and log transforms."""
        df_fe = df.copy()
        
        df_fe['claim_date'] = pd.to_datetime(df_fe['claim_date'], errors='coerce')
        
        if 'claim_date' in df_fe.columns:
             df_fe['claim_year'] = df_fe['claim_date'].dt.year
        
        # Cleaning
        # Changed 2021 to 2025
        df_fe.loc[(df_fe['year_of_born'] < 1900) | (df_fe['year_of_born'] > 2025), 'year_of_born'] = np.nan
        
        # Removed vehicle_made_year cleaning, as vehicle_age is removed
        
        # Log transforms (moved log_annual_income to _engineer_polynomial)
        df_fe['log_claim_est_payout'] = np.log1p(df_fe['claim_est_payout'])
        df_fe['log_vehicle_mileage'] = np.log1p(df_fe['vehicle_mileage'])
        df_fe['log_vehicle_price'] = np.log1p(df_fe['vehicle_price'])
        
        return df_fe

    def _engineer_temporal(self, df):
        """Creates time-based features."""
        df_fe = df.copy()
        
        df_fe['claim_date'] = pd.to_datetime(df_fe['claim_date'], errors='coerce')
        
        if 'claim_date' in df_fe.columns and pd.api.types.is_datetime64_any_dtype(df_fe['claim_date']):
            df_fe['claim_year'] = df_fe['claim_date'].dt.year
            df_fe['claim_month'] = df_fe['claim_date'].dt.month
            df_fe['claim_day'] = df_fe['claim_date'].dt.day
            df_fe['claim_quarter'] = df_fe['claim_date'].dt.quarter
            df_fe['claim_day_of_week'] = df_fe['claim_date'].dt.dayofweek # Renamed from claim_dayofweek
            df_fe['is_weekend'] = (df_fe['claim_day_of_week'] >= 5).astype(int)
            df_fe['is_monday'] = (df_fe['claim_day_of_week'] == 0).astype(int)
            df_fe['is_friday'] = (df_fe['claim_day_of_week'] == 4).astype(int)
            df_fe['is_q4'] = (df_fe['claim_quarter'] == 4).astype(int)

            # --- NEW Features ---
            df_fe['claim_hour'] = df_fe['claim_date'].dt.hour
            df_fe['rush_hour'] = df_fe['claim_hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
            df_fe['late_night'] = df_fe['claim_hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
            # --- End NEW ---

            season_map = {
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall',
                12: 'Winter', 1: 'Winter', 2: 'Winter'
            }
            df_fe['season'] = df_fe['claim_month'].map(season_map).fillna('Unknown')
        
        return df_fe

    def _engineer_polynomial(self, df):
        """Tier 2: Polynomial features."""
        df_fe = df.copy()
        
        # Liability polynomials
        df_fe['liab_prct_squared'] = df_fe['liab_prct'] ** 2
        df_fe['liab_prct_cubed'] = df_fe['liab_prct'] ** 3
        df_fe['liab_prct_sqrt'] = np.sqrt(df_fe['liab_prct'])
        df_fe['liab_prct_log'] = np.log1p(df_fe['liab_prct'])
        df_fe['liab_inverse'] = 100 - df_fe['liab_prct']
        df_fe['liab_inverse_squared'] = (100 - df_fe['liab_prct']) ** 2
        
        # --- NEW Features ---
        # Note: Other log transforms are in _clean_data
        df_fe['log_annual_income'] = np.log1p(df_fe['annual_income'])
        df_fe['sqrt_vehicle_mileage'] = np.sqrt(df_fe['vehicle_mileage'])
        # --- End NEW ---
        
        return df_fe

    def _engineer_accident_evidence(self, df):
        """Accident type & evidence features."""
        df_fe = df.copy()
        
        if 'witness_binary' not in df_fe.columns:
             df_fe['witness_binary'] = (df_fe['witness_present_ind'] == 'Y').astype(int)
        if 'police_binary' not in df_fe.columns:
            df_fe['police_binary'] = df_fe['policy_report_filed_ind']

        df_fe['is_multi_vehicle_clear'] = (df_fe['accident_type'] == 'multi_vehicle_clear').astype(int)
        df_fe['is_multi_vehicle_unclear'] = (df_fe['accident_type'] == 'multi_vehicle_unclear').astype(int)
        df_fe['is_single_car'] = (df_fe['accident_type'] == 'single_car').astype(int)
        
        if 'multicar_binary' not in df_fe.columns:
            df_fe['multicar_binary'] = df_fe['accident_type'].isin(['multi_vehicle_clear', 'multi_vehicle_unclear']).astype(int)
        
        df_fe['has_recovery_target'] = df_fe['multicar_binary']
        df_fe['recovery_case_clarity'] = 0
        df_fe.loc[df_fe['is_multi_vehicle_clear'] == 1, 'recovery_case_clarity'] = 3
        df_fe.loc[df_fe['is_multi_vehicle_unclear'] == 1, 'recovery_case_clarity'] = 1

        df_fe['witness_present'] = df_fe['witness_binary']
        df_fe['police_report'] = df_fe['police_binary']
        
        return df_fe

    def _engineer_accident_site(self, df):
        """Accident site features."""
        df_fe = df.copy()
        
        if 'highrisk_site_binary' not in df_fe.columns:
            df_fe['highrisk_site_binary'] = df_fe['accident_site'].isin(['Highway/Intersection', 'Local']).astype(int)
            
        df_fe['high_risk_site'] = df_fe['highrisk_site_binary']
        df_fe['parking_accident'] = (df_fe['accident_site'] == 'Parking Area').astype(int)
        df_fe['unknown_site'] = (df_fe['accident_site'] == 'Unknown').astype(int)
        df_fe['highway_accident'] = (df_fe['accident_site'] == 'Highway/Intersection').astype(int)
        df_fe['local_accident'] = (df_fe['accident_site'] == 'Local').astype(int)
        
        return df_fe

    def _engineer_driver_age(self, df):
        """Driver age and experience features."""
        df_fe = df.copy()

        if 'claim_year' not in df_fe.columns and 'claim_date' in df_fe.columns:
             df_fe['claim_year'] = pd.to_datetime(df_fe['claim_date'], errors='coerce').dt.year
        
        if 'claim_year' in df_fe.columns:
            df_fe['driver_age'] = df_fe['claim_year'] - df_fe['year_of_born']
            df_fe.loc[(df_fe['driver_age'] < 16) | (df_fe['driver_age'] > 100), 'driver_age'] = np.nan

            df_fe['young_driver'] = ((df_fe['driver_age'] >= 16) & (df_fe['driver_age'] <= 25)).astype(int)
            df_fe['senior_driver'] = (df_fe['driver_age'] > 65).astype(int)
            
            # --- NEW Features ---
            df_fe['prime_driver'] = ((df_fe['driver_age'] > 25) & (df_fe['driver_age'] <= 45)).astype(int)
            df_fe['middle_age_driver'] = ((df_fe['driver_age'] > 45) & (df_fe['driver_age'] <= 65)).astype(int)
            # --- End NEW ---

            df_fe['driving_experience'] = (df_fe['driver_age'] - df_fe['age_of_DL']).clip(lower=0)
            df_fe.loc[df_fe['driving_experience'] < 0, 'driving_experience'] = np.nan
        else:
            cols = ['driver_age', 'young_driver', 'prime_driver', 'middle_age_driver', 'senior_driver', 'driving_experience']
            for col in cols:
                if col not in df_fe.columns:
                    df_fe[col] = np.nan
        
        df_fe['novice_driver'] = (df_fe['driving_experience'] < 3).astype(int)
        df_fe['experienced_driver'] = ((df_fe['driving_experience'] >= 3) & (df_fe['driving_experience'] <= 10)).astype(int)
        df_fe['veteran_driver'] = (df_fe['driving_experience'] > 10).astype(int)

        # --- DELETED Redundant Interactions ---
        # df_fe['experience_x_safety'] = df_fe['driving_experience'] * df_fe['safety_rating']
        # df_fe['safety_x_prior_claims'] = df_fe['safety_rating'] / (1 + df_fe['past_num_of_claims'])
        # df_fe['driver_age_x_safety'] = df_fe['driver_age'] * df_fe['safety_rating']
        # df_fe['young_novice'] = df_fe['young_driver'] * df_fe['novice_driver']
        # --- End DELETED ---
        
        return df_fe
    
    def _engineer_vehicle_features(self, df):
        """Vehicle features. (REMOVED vehicle_age)"""
        df_fe = df.copy()

        df_fe['is_medium_vehicle'] = (df_fe['vehicle_category'] == 'Medium').astype(int)
        df_fe['is_large_vehicle'] = (df_fe['vehicle_category'] == 'Large').astype(int)
        df_fe['is_compact_vehicle'] = (df_fe['vehicle_category'] == 'Compact').astype(int)
        
        return df_fe

    def _engineer_ratios(self, df):
        """Tier 4: Ratio features. (REMOVED mileage_per_year)"""
        df_fe = df.copy()

        # --- 'vehicle_age' dependency and 'mileage_per_year' REMOVED ---
        
        # --- (KEEP) Dependency check for driving_experience ---
        if 'driving_experience' not in df_fe.columns:
            # Recreate minimal dependency logic if needed
            if 'claim_year' not in df_fe.columns and 'claim_date' in df_fe.columns:
                 df_fe['claim_year'] = pd.to_datetime(df_fe['claim_date'], errors='coerce').dt.year
            
            if 'claim_year' in df_fe.columns:
                df_fe['driver_age'] = df_fe['claim_year'] - df_fe['year_of_born']
                df_fe.loc[(df_fe['driver_age'] < 16) | (df_fe['driver_age'] > 100), 'driver_age'] = np.nan
                df_fe['driving_experience'] = (df_fe['driver_age'] - df_fe['age_of_DL']).clip(lower=0)
                df_fe.loc[df_fe['driving_experience'] < 0, 'driving_experience'] = np.nan
            else:
                df_fe['driving_experience'] = np.nan

        # --- (KEEP) Valuable ratio creation ---
        df_fe['payout_to_price_ratio'] = df_fe['claim_est_payout'] / (df_fe['vehicle_price'] + 1)
        
        # --- (DELETE) Redundant manual binning ---
        # df_fe['severe_damage'] = (df_fe['payout_to_price_ratio'] > 0.3).astype(int)
        # df_fe['moderate_damage'] = ((df_fe['payout_to_price_ratio'] >= 0.1) & (df_fe['payout_to_price_ratio'] <= 0.3)).astype(int)
        # df_fe['minor_damage'] = (df_fe['payout_to_price_ratio'] < 0.1).astype(int)

        # --- (KEEP) Valuable ratio creation ---
        df_fe['income_to_price_ratio'] = df_fe['annual_income'] / (df_fe['vehicle_price'] + 1) # Renamed from income_to_vehicle_price
        
        # --- (DELETE) Redundant manual binning ---
        # df_fe['can_afford_vehicle'] = (df_fe['income_to_price_ratio'] >= 0.5).astype(int)
        # df_fe['expensive_for_income'] = (df_fe['income_to_price_ratio'] < 0.3).astype(int)

        # --- (KEEP) Valuable ratio creation ---
        df_fe['claims_per_year_driving'] = df_fe['past_num_of_claims'] / (df_fe['driving_experience'] + 1)
        
        # --- (DELETE) Redundant manual binning ---
        # df_fe['claim_frequency_high'] = (df_fe['claims_per_year_driving'] > 0.5).astype(int)

        # --- (KEEP) Valuable ratio creation ---
        df_fe['safety_to_liability'] = df_fe['safety_rating'] / (df_fe['liab_prct'] + 1)
        df_fe['payout_to_income'] = df_fe['claim_est_payout'] / (df_fe['annual_income'] + 1)
        
        # --- NEW Features ---
        df_fe['mileage_to_price'] = df_fe['vehicle_mileage'] / (df_fe['vehicle_price'] + 1)
        df_fe['weight_to_price'] = df_fe['vehicle_weight'] / (df_fe['vehicle_price'] + 1)
        # --- End NEW ---
        
        return df_fe

    def _engineer_policyholder(self, df):
        """Policyholder characteristics."""
        df_fe = df.copy()

        df_fe['contact_available'] = df_fe['email_or_tel_available']
        df_fe['has_education'] = df_fe['high_education_ind']
        df_fe['recent_move'] = df_fe['address_change_ind']
        df_fe['home_owner'] = (df_fe['living_status'] == 'Own').astype(int)
        df_fe['renter'] = (df_fe['living_status'] == 'Rent').astype(int)
        df_fe['female'] = (df_fe['gender'] == 'F').astype(int)
        
        return df_fe