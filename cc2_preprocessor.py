import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Cecilia and Carol's feature engineering logic encapsulated in a class

class Preprocessor(BaseEstimator, TransformerMixin):
    """
    A sklearn-compatible preprocessor class that encapsulates the 
    feature_engineer function logic.
    
    Can be initialized in 'xgboost' (default) or 'catboost' mode.
    - 'xgboost': Performs target encoding and drops raw categorical columns.
    - 'catboost': Skips target encoding and keeps raw categorical columns.
    
    NOTE: The 'get_state' function for zip_code -> state conversion
    was not provided, so the 'state' feature and any dependent
    target encoding ('state_target_enc') are omitted from this class.
    """
    
    def __init__(self, smoothing_factor=10, mode='xgboost'): 
        self.smoothing_factor = smoothing_factor
        self.mode = mode # <-- ADDED MODE
        self.target_mean_maps_ = {}
        self.global_mean_ = 0
        self.cat_for_encoding_ = [
            'accident_site', 'accident_type', 'channel',
            'vehicle_category', 'vehicle_color', 'living_status',
            'claim_day_of_week', 'gender', 'in_network_bodyshop',
            'season'
        ]

    def fit(self, X, y):
        """
        Learns the target encoding maps from the training data.
        X: DataFrame of features
        y: Series, the target variable (e.g., 'subrogation')
        """
        print(f"Fitting Preprocessor in '{self.mode}' mode...")
        df = X.copy()
        df['subrogation'] = y

        # --- Pre-processing for fitting ---
        # Some features (like 'season') are created here
        # so they can be used in target encoding
        df = self._clean_data(df)
        df = self._engineer_temporal(df)
        
        # --- Learn Target Encodings ---
        if self.mode == 'catboost': # <-- ADDED CHECK
            print("CatBoost mode: Skipping target encoding learning.")
            self.global_mean_ = df['subrogation'].mean() # Still need this for potential fillna
            self.target_mean_maps_ = {}
        else: # Default 'xgboost' behavior
            print("XGBoost mode: Learning target encodings...")
            self.global_mean_ = df['subrogation'].mean()
            self.target_mean_maps_ = {}

            for col in self.cat_for_encoding_:
                if col not in df.columns:
                    print(f"Warning: Column '{col}' not found for target encoding. Skipping.")
                    continue
                    
                # Calculate mean subrogation rate per category
                target_mean = df.groupby(col)['subrogation'].mean()

                # Apply smoothing
                category_counts = df.groupby(col).size()
                smoothing = self.smoothing_factor
                smoothed_mean = (target_mean * category_counts + self.global_mean_ * smoothing) / (category_counts + smoothing)
                
                # Store the map
                self.target_mean_maps_[col] = smoothed_mean

        print("Fit complete.")
        return self
    
    def transform(self, X):
        """
        Applies all feature engineering transformations to the data.
        """
        print(f"Transforming data in '{self.mode}' mode...")
        df_fe = X.copy()

        # Apply all feature engineering steps
        df_fe = self._clean_data(df_fe)
        df_fe = self._engineer_temporal(df_fe)
        # Note: _engineer_geospatial is skipped
        df_fe = self._engineer_critical_interactions(df_fe)
        df_fe = self._engineer_polynomial(df_fe)
        df_fe = self._engineer_accident_evidence(df_fe)
        df_fe = self._engineer_liability_buckets(df_fe)
        df_fe = self._engineer_accident_site(df_fe)
        df_fe = self._engineer_driver_age(df_fe)
        df_fe = self._engineer_vehicle_features(df_fe) # Now modified
        df_fe = self._engineer_claim_characteristics(df_fe)
        df_fe = self._engineer_ratios(df_fe) # Now modified
        df_fe = self._engineer_policyholder(df_fe)
        df_fe = self._engineer_channel(df_fe)
        df_fe = self._engineer_composite_scores(df_fe)
        df_fe = self._engineer_domain_flags(df_fe)
        
        # --- ADDED CALLS TO NEW METHODS ---
        df_fe = self._engineer_vehicle_proxies(df_fe)
        df_fe = self._engineer_statistical_features(df_fe)
        df_fe = self._engineer_hybrid_indices(df_fe)
        # --- END ADDITION ---

        # --- Apply Learned Target Encodings ---
        if self.mode == 'catboost':
            print("CatBoost mode: Skipping target encoding application.")
        else:
            print("XGBoost mode: Dropping final object/datetime columns.")
            cols_to_drop = df_fe.select_dtypes(include=['object', 'datetime64[ns]']).columns
            if len(cols_to_drop) > 0:
                print(f"Dropping final object/datetime columns: {list(cols_to_drop)}")
                df_fe = df_fe.drop(columns=cols_to_drop)

        print("Transform complete.")
        return df_fe

    # =======================================================================
    # HELPER METHODS (from feature_engineer function)
    # =======================================================================
    
    def _clean_data(self, df):
        """Applies data cleaning."""
        df_fe = df.copy()
        
        # Ensure claim_date is datetime for other functions
        df_fe['claim_date'] = pd.to_datetime(df_fe['claim_date'], errors='coerce')
        
        # Need claim_year for other functions
        if 'claim_date' in df_fe.columns:
             df_fe['claim_year'] = df_fe['claim_date'].dt.year
        
        # Cleaning
        df_fe.loc[(df_fe['year_of_born'] < 1900) | (df_fe['year_of_born'] > 2025), 'year_of_born'] = np.nan
        
        # --- REMOVED vehicle_made_year cleaning block ---

        # Log transforms moved to _engineer_polynomial
        
        return df_fe

    def _engineer_temporal(self, df):
        """Creates time-based features."""
        df_fe = df.copy()
        
        # Ensure 'claim_date' is datetime (might be redundant if _clean_data ran)
        df_fe['claim_date'] = pd.to_datetime(df_fe['claim_date'], errors='coerce')
        
        if 'claim_date' in df_fe.columns and pd.api.types.is_datetime64_any_dtype(df_fe['claim_date']):
            df_fe['claim_year'] = df_fe['claim_date'].dt.year
            df_fe['claim_month'] = df_fe['claim_date'].dt.month
            df_fe['claim_day'] = df_fe['claim_date'].dt.day
            df_fe['claim_quarter'] = df_fe['claim_date'].dt.quarter
            df_fe['claim_day_of_week'] = df_fe['claim_date'].dt.dayofweek
            df_fe['is_weekend'] = (df_fe['claim_day_of_week'] >= 5).astype(int)
            df_fe['is_monday'] = (df_fe['claim_day_of_week'] == 0).astype(int)
            df_fe['is_friday'] = (df_fe['claim_day_of_week'] == 4).astype(int)
            df_fe['is_q4'] = (df_fe['claim_quarter'] == 4).astype(int)

            season_map = {
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall',
                12: 'Winter', 1: 'Winter', 2: 'Winter'
            }
            df_fe['season'] = df_fe['claim_month'].map(season_map).fillna('Unknown')
        
        return df_fe

    # def _engineer_geospatial(self, df):
    #     """
    #     Creates state feature from zip code.
    #     OMITTED because get_state() function is not defined.
    #     """
    #     df_fe = df.copy()
    #     # df_fe['state'] = df_fe['zip_code'].apply(get_state)
    #     return df_fe
    
    def _engineer_critical_interactions(self, df):
        """Tier 1: Critical interaction features."""
        df_fe = df.copy()
        
        df_fe['witness_binary'] = (df_fe['witness_present_ind'] == 'Y').astype(int)
        df_fe['police_binary'] = df_fe['policy_report_filed_ind']
        df_fe['multicar_binary'] = df_fe['accident_type'].isin(['multi_vehicle_clear', 'multi_vehicle_unclear']).astype(int)
        df_fe['highrisk_site_binary'] = df_fe['accident_site'].isin(['Highway/Intersection', 'Local']).astype(int)

        liab_prct_filled = df_fe['liab_prct'].fillna(0)
        df_fe['liab_x_witness'] = liab_prct_filled * df_fe['witness_binary']
        df_fe['liab_x_police'] = liab_prct_filled * df_fe['police_binary']
        df_fe['liab_x_multicar'] = liab_prct_filled * df_fe['multicar_binary']
        df_fe['liab_x_highrisk_site'] = liab_prct_filled * df_fe['highrisk_site_binary']
        df_fe['liab_x_evidence'] = liab_prct_filled * (df_fe['witness_binary'] + df_fe['police_binary'])

        df_fe['witness_x_police'] = df_fe['witness_binary'] * df_fe['police_binary']
        df_fe['witness_x_multicar'] = df_fe['witness_binary'] * df_fe['multicar_binary']
        df_fe['police_x_multicar'] = df_fe['police_binary'] * df_fe['multicar_binary']
        df_fe['multicar_x_highrisk'] = df_fe['multicar_binary'] * df_fe['highrisk_site_binary']
        
        # --- ADDED FROM NEW SCRIPT ---
        df_fe['liab_x_payout'] = liab_prct_filled * df_fe['claim_est_payout']
        df_fe['liab_x_mileage'] = liab_prct_filled * df_fe['vehicle_mileage']

        # Note: 'claim_day_of_week' is created in _engineer_temporal
        if 'claim_day_of_week' in df_fe.columns and 'accident_site' in df_fe.columns:
            df_fe['weekend_highway'] = (df_fe['claim_day_of_week'] >= 5).astype(int) * (df_fe['accident_site'] == 'Highway/Intersection').astype(int)
        
        df_fe['witness_police_multicar'] = df_fe['witness_binary'] * df_fe['police_binary'] * df_fe['multicar_binary']
        # --- END ADDITION ---

        return df_fe
    
    def _engineer_polynomial(self, df):
        """Tier 2: Polynomial & Log/Sqrt features."""
        df_fe = df.copy()
        
        # Liability Polynomials
        df_fe['liab_prct_squared'] = df_fe['liab_prct'] ** 2
        df_fe['liab_prct_cubed'] = df_fe['liab_prct'] ** 3
        df_fe['liab_prct_sqrt'] = np.sqrt(df_fe['liab_prct'])
        df_fe['liab_prct_log'] = np.log1p(df_fe['liab_prct'])
        df_fe['liab_inverse'] = 100 - df_fe['liab_prct']
        df_fe['liab_inverse_squared'] = (100 - df_fe['liab_prct']) ** 2
        
        # --- ADDED/MOVED FROM NEW SCRIPT ---
        # Log transforms (moved from _clean_data)
        df_fe['log_claim_est_payout'] = np.log1p(df_fe['claim_est_payout'])
        df_fe['log_vehicle_mileage'] = np.log1p(df_fe['vehicle_mileage'])
        df_fe['log_vehicle_price'] = np.log1p(df_fe['vehicle_price'])
        
        # New transforms
        df_fe['log_annual_income'] = np.log1p(df_fe['annual_income'])
        df_fe['sqrt_vehicle_mileage'] = np.sqrt(df_fe['vehicle_mileage'])
        # --- END ADDITION ---

        return df_fe

    def _engineer_accident_evidence(self, df):
        """Accident type & evidence features."""
        df_fe = df.copy()
        
        # Ensure dependencies are created if not present
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
        df_fe['evidence_none'] = ((df_fe['witness_present'] == 0) & (df_fe['police_report'] == 0)).astype(int)
        df_fe['evidence_weak'] = (((df_fe['witness_present'] == 1) & (df_fe['police_report'] == 0)) |
                               ((df_fe['witness_present'] == 0) & (df_fe['police_report'] == 1))).astype(int)
        df_fe['evidence_strong'] = ((df_fe['witness_present'] == 1) & (df_fe['police_report'] == 1)).astype(int)
        df_fe['evidence_very_strong'] = ((df_fe['witness_present'] == 1) & (df_fe['police_report'] == 1) &
                                      (df_fe['liab_prct'] < 20)).astype(int)
        df_fe['evidence_score'] = df_fe['witness_present'] + df_fe['police_report']
        
        return df_fe

    def _engineer_liability_buckets(self, df):
        """Liability bucket features."""
        df_fe = df.copy()
        
        df_fe['liab_under_10'] = (df_fe['liab_prct'] < 10).astype(int)
        df_fe['liab_10_to_15'] = ((df_fe['liab_prct'] >= 10) & (df_fe['liab_prct'] < 15)).astype(int)
        df_fe['liab_15_to_20'] = ((df_fe['liab_prct'] >= 15) & (df_fe['liab_prct'] < 20)).astype(int)
        df_fe['liab_20_to_25'] = ((df_fe['liab_prct'] >= 20) & (df_fe['liab_prct'] < 25)).astype(int)
        df_fe['liab_25_to_30'] = ((df_fe['liab_prct'] >= 25) & (df_fe['liab_prct'] < 30)).astype(int)
        df_fe['liab_30_to_35'] = ((df_fe['liab_prct'] >= 30) & (df_fe['liab_prct'] < 35)).astype(int)
        df_fe['liab_35_to_40'] = ((df_fe['liab_prct'] >= 35) & (df_fe['liab_prct'] < 40)).astype(int)
        df_fe['liab_40_to_50'] = ((df_fe['liab_prct'] >= 40) & (df_fe['liab_prct'] < 50)).astype(int)
        df_fe['liab_over_50'] = (df_fe['liab_prct'] >= 50).astype(int)
        df_fe['not_at_fault'] = df_fe['liab_under_10']
        df_fe['minimal_fault'] = (df_fe['liab_prct'] < 25).astype(int)
        df_fe['low_fault'] = (df_fe['liab_prct'] < 35).astype(int)
        df_fe['shared_fault'] = ((df_fe['liab_prct'] >= 35) & (df_fe['liab_prct'] < 50)).astype(int)
        df_fe['high_fault'] = (df_fe['liab_prct'] >= 50).astype(int)
        
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
            # --- ADDED BUCKETS ---
            df_fe['prime_driver'] = ((df_fe['driver_age'] > 25) & (df_fe['driver_age'] <= 45)).astype(int)
            df_fe['middle_age_driver'] = ((df_fe['driver_age'] > 45) & (df_fe['driver_age'] <= 65)).astype(int)
            # --- END ADDITION ---
            df_fe['senior_driver'] = (df_fe['driver_age'] > 65).astype(int)

            df_fe['driving_experience'] = (df_fe['driver_age'] - df_fe['age_of_DL']).clip(lower=0)
            df_fe.loc[df_fe['driving_experience'] < 0, 'driving_experience'] = np.nan
        else:
            # Create NaNs if claim_year couldn't be created
            cols = ['driver_age', 'young_driver', 'prime_driver', 'middle_age_driver', 'senior_driver', 'driving_experience']
            for col in cols:
                if col not in df_fe.columns:
                    df_fe[col] = np.nan
        
        df_fe['novice_driver'] = (df_fe['driving_experience'] < 3).astype(int)
        df_fe['experienced_driver'] = ((df_fe['driving_experience'] >= 3) & (df_fe['driving_experience'] <= 10)).astype(int)
        df_fe['veteran_driver'] = (df_fe['driving_experience'] > 10).astype(int)

        df_fe['experience_x_safety'] = df_fe['driving_experience'] * df_fe['safety_rating']
        df_fe['safety_x_prior_claims'] = df_fe['safety_rating'] / (1 + df_fe['past_num_of_claims'])

        # --- ADDED FROM NEW SCRIPT ---
        df_fe['driver_age_x_safety'] = df_fe['driver_age'] * df_fe['safety_rating']
        # --- END ADDITION ---
        
        return df_fe
    
    def _engineer_vehicle_features(self, df):
        """Vehicle features (without vehicle_age)."""
        df_fe = df.copy()
        
        # --- REMOVED vehicle_age, new_vehicle, mid_age_vehicle, old_vehicle ---
        # --- REMOVED claim_year dependency logic ---

        df_fe['luxury_vehicle'] = (df_fe['vehicle_price'] > 50000).astype(int)
        df_fe['mid_price_vehicle'] = ((df_fe['vehicle_price'] >= 20000) & (df_fe['vehicle_price'] <= 50000)).astype(int)
        df_fe['economy_vehicle'] = (df_fe['vehicle_price'] < 20000).astype(int)

        df_fe['heavy_vehicle'] = (df_fe['vehicle_weight'] > 30000).astype(int)
        df_fe['light_vehicle'] = (df_fe['vehicle_weight'] < 15000).astype(int)
        # --- ADDED FROM NEW SCRIPT ---
        df_fe['medium_weight'] = ((df_fe['vehicle_weight'] >= 15000) & (df_fe['vehicle_weight'] <= 30000)).astype(int)
        # --- END ADDITION ---

        df_fe['is_large_vehicle'] = (df_fe['vehicle_category'] == 'Large').astype(int)
        df_fe['is_compact_vehicle'] = (df_fe['vehicle_category'] == 'Compact').astype(int)
        # --- ADDED FROM NEW SCRIPT ---
        df_fe['is_medium_vehicle'] = (df_fe['vehicle_category'] == 'Medium').astype(int)
        # --- END ADDITION ---
        
        return df_fe
    
    def _engineer_claim_characteristics(self, df):
        """Claim characteristics."""
        df_fe = df.copy()
        
        df_fe['high_mileage'] = (df_fe['vehicle_mileage'] > 100000).astype(int)
        df_fe['low_mileage'] = (df_fe['vehicle_mileage'] < 50000).astype(int)
        df_fe['very_high_mileage'] = (df_fe['vehicle_mileage'] > 150000).astype(int)
        # --- ADDED FROM NEW SCRIPT ---
        df_fe['medium_mileage'] = ((df_fe['vehicle_mileage'] >= 50000) & (df_fe['vehicle_mileage'] <= 100000)).astype(int)
        # --- END ADDITION ---

        df_fe['frequent_claimer'] = (df_fe['past_num_of_claims'] > 5).astype(int)
        df_fe['moderate_claimer'] = ((df_fe['past_num_of_claims'] >= 1) & (df_fe['past_num_of_claims'] <= 5)).astype(int)
        df_fe['first_time_claimer'] = (df_fe['past_num_of_claims'] == 0).astype(int)
        df_fe['very_frequent_claimer'] = (df_fe['past_num_of_claims'] > 10).astype(int)

        df_fe['large_payout'] = (df_fe['claim_est_payout'] > 5000).astype(int)
        df_fe['medium_payout'] = ((df_fe['claim_est_payout'] >= 2000) & (df_fe['claim_est_payout'] <= 5000)).astype(int)
        df_fe['small_payout'] = (df_fe['claim_est_payout'] < 2000).astype(int)
        df_fe['very_large_payout'] = (df_fe['claim_est_payout'] > 8000).astype(int)
        
        # --- ADDED FROM NEW SCRIPT ---
        df_fe['mileage_x_claims'] = df_fe['vehicle_mileage'] * df_fe['past_num_of_claims']
        # --- END ADDITION ---

        return df_fe
    
    def _engineer_ratios(self, df):
        """Tier 4: Ratio features."""
        df_fe = df.copy()

        # --- REMOVED dependency checks for vehicle_age ---
        
        # Ensure driving_experience exists for claims_per_year_driving
        if 'driving_experience' not in df_fe.columns:
            if 'claim_year' not in df_fe.columns and 'claim_date' in df_fe.columns:
                 df_fe['claim_year'] = pd.to_datetime(df_fe['claim_date'], errors='coerce').dt.year
            
            if 'claim_year' in df_fe.columns:
                df_fe['driver_age'] = df_fe['claim_year'] - df_fe['year_of_born']
                df_fe.loc[(df_fe['driver_age'] < 16) | (df_fe['driver_age'] > 100), 'driver_age'] = np.nan
                df_fe['driving_experience'] = (df_fe['driver_age'] - df_fe['age_of_DL']).clip(lower=0)
                df_fe.loc[df_fe['driving_experience'] < 0, 'driving_experience'] = np.nan
            else:
                df_fe['driving_experience'] = np.nan


        df_fe['payout_to_price_ratio'] = df_fe['claim_est_payout'] / (df_fe['vehicle_price'] + 1)
        df_fe['severe_damage'] = (df_fe['payout_to_price_ratio'] > 0.3).astype(int)
        df_fe['moderate_damage'] = ((df_fe['payout_to_price_ratio'] >= 0.1) & (df_fe['payout_to_price_ratio'] <= 0.3)).astype(int)
        df_fe['minor_damage'] = (df_fe['payout_to_price_ratio'] < 0.1).astype(int)

        # Renamed from income_to_price_ratio to match new script
        df_fe['income_to_vehicle_price'] = df_fe['annual_income'] / (df_fe['vehicle_price'] + 1)
        df_fe['can_afford_vehicle'] = (df_fe['income_to_vehicle_price'] >= 0.5).astype(int)
        df_fe['expensive_for_income'] = (df_fe['income_to_vehicle_price'] < 0.3).astype(int)

        # --- REMOVED mileage_per_year and high_annual_mileage ---

        df_fe['claims_per_year_driving'] = df_fe['past_num_of_claims'] / (df_fe['driving_experience'] + 1)
        df_fe['claim_frequency_high'] = (df_fe['claims_per_year_driving'] > 0.5).astype(int)

        df_fe['safety_to_liability'] = df_fe['safety_rating'] / (df_fe['liab_prct'] + 1)
        df_fe['payout_to_income'] = df_fe['claim_est_payout'] / (df_fe['annual_income'] + 1)
        
        # --- ADDED FROM NEW SCRIPT ---
        df_fe['mileage_to_price'] = df_fe['vehicle_mileage'] / (df_fe['vehicle_price'] + 1)
        df_fe['weight_to_price'] = df_fe['vehicle_weight'] / (df_fe['vehicle_price'] + 1)
        # --- END ADDITION ---

        return df_fe
    
    def _engineer_policyholder(self, df):
        """Policyholder characteristics."""
        df_fe = df.copy()
        
        # ... (income buckets are unchanged) ...
        df_fe['high_income'] = (df_fe['annual_income'] > 70000).astype(int)
        df_fe['mid_income'] = ((df_fe['annual_income'] >= 40000) & (df_fe['annual_income'] <= 70000)).astype(int)
        df_fe['low_income'] = (df_fe['annual_income'] < 40000).astype(int)
        df_fe['very_high_income'] = (df_fe['annual_income'] > 100000).astype(int)

        df_fe['high_safety_rating'] = (df_fe['safety_rating'] > 80).astype(int)
        df_fe['low_safety_rating'] = (df_fe['safety_rating'] < 60).astype(int)
        df_fe['very_high_safety'] = (df_fe['safety_rating'] > 90).astype(int)
        # --- ADDED FROM NEW SCRIPT ---
        df_fe['medium_safety'] = ((df_fe['safety_rating'] >= 60) & (df_fe['safety_rating'] <= 80)).astype(int)
        # --- END ADDITION ---

        # ... (rest of the features are unchanged) ...
        df_fe['contact_available'] = df_fe['email_or_tel_available']
        df_fe['has_education'] = df_fe['high_education_ind']
        df_fe['recent_move'] = df_fe['address_change_ind']
        df_fe['home_owner'] = (df_fe['living_status'] == 'Own').astype(int)
        df_fe['renter'] = (df_fe['living_status'] == 'Rent').astype(int)
        df_fe['female'] = (df_fe['gender'] == 'F').astype(int)
        
        return df_fe
    
    def _engineer_channel(self, df):
        """Channel features."""
        df_fe = df.copy()
        
        df_fe['via_broker'] = (df_fe['channel'] == 'Broker').astype(int)
        df_fe['via_online'] = (df_fe['channel'] == 'Online').astype(int)
        df_fe['via_phone'] = (df_fe['channel'] == 'Phone').astype(int)
        df_fe['in_network_repair'] = (df_fe['in_network_bodyshop'] == 'yes').astype(int)
        # --- ADDED FROM NEW SCRIPT ---
        df_fe['out_network_repair'] = (df_fe['in_network_bodyshop'] == 'no').astype(int)
        # --- END ADDITION ---

        return df_fe
    
    def _engineer_composite_scores(self, df):
        """Composite scores."""
        df_fe = df.copy()

        # ... (dependencies and recovery_feasibility_score are unchanged) ...
        if 'evidence_none' not in df_fe.columns:
            df_fe = self._engineer_accident_evidence(df_fe)
        if 'high_risk_site' not in df_fe.columns:
            df_fe = self._engineer_accident_site(df_fe)

        liability_score = np.sqrt((100 - df_fe['liab_prct']) / 100.0)
        evidence_score = (df_fe['evidence_none'] * 0.0 + df_fe['evidence_weak'] * 0.4 +
                          df_fe['evidence_strong'] * 0.7 + df_fe['evidence_very_strong'] * 1.0)
        clarity_score = df_fe['recovery_case_clarity'] / 3.0
        site_score = df_fe['high_risk_site'] * 0.7 + (1 - df_fe['unknown_site']) * 0.3

        df_fe['recovery_feasibility_score'] = (0.35 * liability_score + 0.30 * df_fe['has_recovery_target'] +
                                             0.20 * evidence_score + 0.10 * clarity_score + 0.05 * site_score)
        
        # --- ADDED FROM NEW SCRIPT ---
        df_fe['case_confidence_score'] = (
            0.4 * (100 - df_fe['liab_prct']) / 100 +
            0.4 * df_fe['evidence_score'] / 2 +
            0.2 * df_fe['recovery_case_clarity'] / 3
        )
        # --- END ADDITION ---

        return df_fe
    
    def _engineer_domain_flags(self, df):
        """Domain logic flags."""
        df_fe = df.copy()
        
        # ... (dependencies and existing flags are unchanged) ...
        if 'evidence_strong' not in df_fe.columns:
            df_fe = self._engineer_accident_evidence(df_fe)
        if 'high_risk_site' not in df_fe.columns:
            df_fe = self._engineer_accident_site(df_fe)

        df_fe['perfect_case'] = ((df_fe['liab_prct'] < 15) & (df_fe['witness_present'] == 1) &
                              (df_fe['police_report'] == 1) & (df_fe['has_recovery_target'] == 1)).astype(int)
        # ... (strong_case, good_case, weak_case, no_case, high_value_opportunity, slam_dunk_case) ...
        df_fe['strong_case'] = ((df_fe['liab_prct'] < 25) & (df_fe['evidence_strong'] == 1) &
                             (df_fe['has_recovery_target'] == 1)).astype(int)
        df_fe['good_case'] = ((df_fe['liab_prct'] < 35) & (df_fe['evidence_score'] >= 1) &
                           (df_fe['has_recovery_target'] == 1)).astype(int)
        df_fe['weak_case'] = ((df_fe['liab_prct'] > 40) | (df_fe['is_single_car'] == 1) |
                           (df_fe['evidence_none'] == 1)).astype(int)
        df_fe['no_case'] = ((df_fe['liab_prct'] > 60) | ((df_fe['is_single_car'] == 1) & (df_fe['evidence_none'] == 1))).astype(int)
        df_fe['high_value_opportunity'] = ((df_fe['claim_est_payout'] > 3000) & (df_fe['liab_prct'] < 30) &
                                         (df_fe['has_recovery_target'] == 1)).astype(int)
        
        if 'multicar_binary' not in df_fe.columns:
            df_fe['multicar_binary'] = df_fe['accident_type'].isin(['multi_vehicle_clear', 'multi_vehicle_unclear']).astype(int)

        df_fe['slam_dunk_case'] = ((df_fe['liab_prct'] < 10) & (df_fe['witness_present'] == 1) &
                                (df_fe['police_report'] == 1) & (df_fe['multicar_binary'] == 1) &
                                (df_fe['high_risk_site'] == 1)).astype(int)

        # --- ADDED FROM NEW SCRIPT ---
        df_fe['low_liab_high_payout'] = ((df_fe['liab_prct'] < 20) & (df_fe['claim_est_payout'] > 5000)).astype(int)
        df_fe['clear_fault_case'] = ((df_fe['liab_prct'] < 15) & (df_fe['multicar_binary'] == 1)).astype(int)
        df_fe['high_mileage_low_fault'] = ((df_fe['vehicle_mileage'] > 100000) & (df_fe['liab_prct'] < 30)).astype(int)

        # Temporal & Behavior Dynamics
        df_fe['claim_early_in_year'] = (df_fe['claim_month'] <= 3).astype(int)
        df_fe['claim_end_of_year'] = (df_fe['claim_month'] >= 10).astype(int)
        df_fe['weekend_parking'] = df_fe['is_weekend'] * (df_fe['accident_site'] == 'Parking Area').astype(int)
        df_fe['winter_claim_high_payout'] = ((df_fe['season'] == 'Winter') & (df_fe['claim_est_payout'] > 5000)).astype(int)

        # Policyholder Risk Profile
        if 'renter' not in df_fe.columns: # Dependency
            df_fe['renter'] = (df_fe['living_status'] == 'Rent').astype(int)
        if 'expensive_for_income' not in df_fe.columns: # Dependency
            df_fe['income_to_vehicle_price'] = df_fe['annual_income'] / (df_fe['vehicle_price'] + 1)
            df_fe['expensive_for_income'] = (df_fe['income_to_vehicle_price'] < 0.3).astype(int)

        df_fe['unstable_policyholder'] = ((df_fe['recent_move'] == 1) & (df_fe['renter'] == 1)).astype(int)
        df_fe['financial_stress_risk'] = ((df_fe['expensive_for_income'] == 1) & (df_fe['large_payout'] == 1)).astype(int)
        df_fe['young_driver_highway'] = df_fe['young_driver'] * df_fe['highway_accident']
        df_fe['senior_driver_parking'] = df_fe['senior_driver'] * df_fe['parking_accident']

        # Liability & Evidence Interaction
        df_fe['low_liab_weak_evidence'] = ((df_fe['liab_prct'] < 20) & (df_fe['evidence_weak'] == 1)).astype(int)
        df_fe['high_liab_strong_evidence'] = ((df_fe['liab_prct'] > 50) & (df_fe['evidence_strong'] == 1)).astype(int)
        # --- END ADDITION ---

        return df_fe
    
    def _engineer_vehicle_proxies(self, df):
        """Vehicle utilization proxies (replaces vehicle_age)."""
        df_fe = df.copy()
        
        df_fe['mileage_x_weight'] = df_fe['vehicle_mileage'] * df_fe['vehicle_weight']
        df_fe['mileage_per_dollar'] = df_fe['vehicle_mileage'] / (df_fe['vehicle_price'] + 1)
        df_fe['payout_to_weight'] = df_fe['claim_est_payout'] / (df_fe['vehicle_weight'] + 1)
        
        return df_fe

    def _engineer_statistical_features(self, df):
        """Statistical Normalization & Percentile Features."""
        df_fe = df.copy()

        # Z-scores
        for col in ['claim_est_payout', 'vehicle_mileage', 'annual_income']:
            if col in df_fe.columns:
                mean = df_fe[col].mean()
                std = df_fe[col].std() + 1e-9 # Avoid division by zero
                df_fe[f'{col}_z'] = (df_fe[col] - mean) / std

        # Percentiles
        try:
            df_fe['liab_percentile'] = pd.qcut(df_fe['liab_prct'], 10, labels=False, duplicates='drop')
            df_fe['payout_percentile'] = pd.qcut(df_fe['claim_est_payout'], 10, labels=False, duplicates='drop')
        except Exception:
            # Handle cases with insufficient unique values
            df_fe['liab_percentile'] = np.nan
            df_fe['payout_percentile'] = np.nan
            
        return df_fe

    def _engineer_hybrid_indices(self, df):
        """Aggregate / Hybrid Indices."""
        df_fe = df.copy()

        # Ensure dependencies
        if 'evidence_score' not in df_fe.columns:
            df_fe = self._engineer_accident_evidence(df_fe)
        if 'claims_per_year_driving' not in df_fe.columns:
            df_fe = self._engineer_ratios(df_fe) # This creates it

        df_fe['case_strength_index'] = df_fe['evidence_score'] * (1 - df_fe['liab_prct'] / 100)
        
        df_fe['financial_exposure_index'] = (
            (df_fe['claim_est_payout'] / (df_fe['annual_income'] + 1)) * (1 + df_fe['liab_prct'] / 100)
        )
        
        df_fe['behavioral_risk_index'] = (
            df_fe['claims_per_year_driving'] * (100 - df_fe['safety_rating']) / 100
        )
        
        return df_fe