import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self):
        self.id_column = ['claim_number']
        self.min_driver_age = 14
        self.max_driver_age = 120

        self.categorical_cols_ = None
        self.label_encoders = {}
        self.income_q25 = None
        self.income_q75 = None

    def _coerce_and_clean(self, df):
        df = df.copy()
        df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
        df['claim_year'] = df['claim_date'].dt.year
        df.loc[(df['year_of_born'] < 1900) | (df['year_of_born'] > 2025), 'year_of_born'] = np.nan
        future_mask = df['vehicle_made_year'] > df['claim_year']
        df.loc[future_mask, 'vehicle_made_year'] = np.nan
        return df

    def _engineer_accident_features(self, df):
        df['is_multi_vehicle_clear'] = (df['accident_type'] == 'multi_vehicle_clear').astype(int)
        df['is_multi_vehicle_unclear'] = (df['accident_type'] == 'multi_vehicle_unclear').astype(int)
        df['is_single_car'] = (df['accident_type'] == 'single_car').astype(int)
        df['has_recovery_target'] = (df['is_multi_vehicle_clear'] | df['is_multi_vehicle_unclear']).astype(int)
        df['recovery_case_clarity'] = 0
        df.loc[df['is_multi_vehicle_clear'] == 1, 'recovery_case_clarity'] = 3
        df.loc[df['is_multi_vehicle_unclear'] == 1, 'recovery_case_clarity'] = 1
        return df

    def _engineer_evidence_features(self, df):
        df['witness_present'] = df['witness_present_ind'].map({'Y': 1, 'N': 0})
        df['evidence_none'] = ((df['witness_present'] == 0) & (df['policy_report_filed_ind'] == 0)).astype(int)
        df['evidence_weak'] = (
            ((df['witness_present'] == 1) & (df['policy_report_filed_ind'] == 0)) |
            ((df['witness_present'] == 0) & (df['policy_report_filed_ind'] == 1))
        ).astype(int)
        df['evidence_strong'] = ((df['witness_present'] == 1) & (df['policy_report_filed_ind'] == 1)).astype(int)
        df['evidence_very_strong'] = (
            (df['witness_present'] == 1) &
            (df['policy_report_filed_ind'] == 1) &
            (df['liab_prct'] < 20)
        ).astype(int)
        
        # Bin liability
        df['not_at_fault'] = (df['liab_prct'] < 10).astype(int)
        df['minimal_fault'] = ((df['liab_prct'] >= 10) & (df['liab_prct'] <= 20)).astype(int)
        df['shared_fault'] = (df['liab_prct'] > 20).astype(int)
        return df

    def _engineer_driver_features(self, df):
        df['driver_age'] = df['claim_year'] - df['year_of_born']
        bad_age = (df['driver_age'] < self.min_driver_age) | (df['driver_age'] > self.max_driver_age)
        df.loc[bad_age, 'driver_age'] = np.nan
        
        # Bin driver age
        df['young_driver_18_25'] = ((df['driver_age'] >= 18) & (df['driver_age'] <= 25)).astype(int)
        df['adult_driver_26_45'] = ((df['driver_age'] >= 26) & (df['driver_age'] <= 45)).astype(int)
        df['middle_age_driver_46_65'] = ((df['driver_age'] >= 46) & (df['driver_age'] <= 65)).astype(int)
        df['senior_driver_65plus'] = (df['driver_age'] > 65).astype(int)

        # Bin driving experience
        df['driving_experience'] = df['driver_age'] - df['age_of_DL']
        df.loc[df['driving_experience'] < 0, 'driving_experience'] = np.nan
        df['novice_driver'] = (df['driving_experience'] < 2).astype(int)
        df['experienced_2_5y'] = ((df['driving_experience'] >= 2) & (df['driving_experience'] <= 5)).astype(int)
        df['experienced_5_10y'] = ((df['driving_experience'] > 5) & (df['driving_experience'] <= 10)).astype(int)
        df['veteran_driver'] = (df['driving_experience'] > 10).astype(int)
        return df

    def _engineer_channel_features(self, df):
        """Creates features for the claim channel."""
        df['via_broker'] = (df['channel'] == 'Broker').astype(int)
        df['via_online'] = (df['channel'] == 'Online').astype(int)
        df['via_phone']  = (df['channel'] == 'Phone').astype(int)
        df['channel_good_documentation'] = df['channel'].isin(['Broker', 'Online']).astype(int)
        return df

    def _engineer_policyholder_features(self, df):
        """Creates features for policyholder info."""
        # Initialize income bins (will be filled in transform)
        df['low_income'] = np.nan
        df['middle_income'] = np.nan
        df['high_income'] = np.nan

        df['has_high_education'] = df['high_education_ind']
        df['recent_address_change'] = df['address_change_ind']
        df['home_owner'] = (df['living_status'] == 'Own').astype(int)
        df['renter'] = (df['living_status'] == 'Rent').astype(int)
        df['contact_info_available'] = df['email_or_tel_available']
        df['in_network_repair'] = (df['in_network_bodyshop'] == 'yes').astype(int)
        df['out_of_network_repair'] = (df['in_network_bodyshop'] == 'no').astype(int)
        return df

    def _engineer_domain_scores(self, df):
        liability_score = np.sqrt((100 - df['liab_prct']) / 100.0)
        evidence_score  = (df['evidence_none'] * 0.0 +
                           df['evidence_weak'] * 0.5 +
                           df['evidence_strong'] * 0.8 +
                           df['evidence_very_strong'] * 1.0)
        clarity_score = df['recovery_case_clarity'] / 3.0
        info_score = df['channel_good_documentation'] * 0.7 + df['contact_info_available'] * 0.3

        weights = np.array([0.30, 0.30, 0.20, 0.15, 0.05])
        parts = np.vstack([
            liability_score,
            df['has_recovery_target'],
            evidence_score,
            clarity_score,
            info_score
        ])
        df['recovery_feasibility_score'] = (parts * weights.reshape(-1,1)).sum(axis=0)

        # Final potential flags
        df['high_subrogation_potential'] = (
            (df['liab_prct'] < 20) &
            (df['has_recovery_target'] == 1) &
            (df['evidence_strong'] == 1) &
            (df['recovery_feasibility_score'] > 0.7)
        ).astype(int)
        df['likely_no_subrogation'] = (
            (df['liab_prct'] > 50) |
            (df['is_single_car'] == 1) |
            (df['evidence_none'] == 1)
        ).astype(int)
        df['potential_subrogation_case'] = (df['high_subrogation_potential'] == 1).astype(int)
        return df

    def _create_features(self, df):
        df = df.copy()
        
        df = self._engineer_accident_features(df)
        df = self._engineer_evidence_features(df)
        df = self._engineer_driver_features(df)
        df = self._engineer_channel_features(df)
        df = self._engineer_policyholder_features(df)
        df = self._engineer_domain_scores(df)
        
        df = df.drop(columns=self.id_column, errors='ignore')
        return df

    def fit(self, df):
        df_clean = self._coerce_and_clean(df.copy())
        
        # Learn quantiles
        self.income_q25 = df_clean['annual_income'].quantile(0.25)
        self.income_q75 = df_clean['annual_income'].quantile(0.75)

        # Learn categories for label encoding
        self.categorical_cols_ = list(df_clean.select_dtypes(include=['object']).columns)
        self.label_encoders.clear()
        for col in self.categorical_cols_:
            le = LabelEncoder()
            # Fit on the original string columns
            le.fit(df_clean[col].astype(str)) 
            self.label_encoders[col] = le
            
        return self # Fit is done

    def transform(self, df):
        # Clean the data
        df_clean = self._coerce_and_clean(df.copy())
        
        # Create all the new features
        df_features = self._create_features(df_clean)
        
        # Apply learned quantiles
        q25, q75 = self.income_q25, self.income_q75
        df_features['low_income']    = (df_features['annual_income'] <= q25).astype(int)
        df_features['middle_income'] = ((df_features['annual_income'] > q25) & (df_features['annual_income'] <= q75)).astype(int)
        df_features['high_income']   = (df_features['annual_income'] > q75).astype(int)

        # Apply learned encoders
        for col, le in self.label_encoders.items():
            if col in df_features.columns:
                df_features[col] = le.transform(df_features[col].astype(str))

        # Final cleanup
        df_features = df_features.drop(columns=['claim_date'], errors='ignore')
        return df_features

    def fit_transform(self, df):
        return self.fit(df).transform(df)