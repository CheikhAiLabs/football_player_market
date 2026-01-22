import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import joblib

class FootballDataPipeline:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.label_encoders = {}
        
    def load_data(self):
        print("Loading datasets...")
        
        self.player_profiles = pd.read_csv(
            f'{self.data_dir}/player_profiles/player_profiles.csv',
            low_memory=False
        )
        self.player_market_value = pd.read_csv(
            f'{self.data_dir}/player_market_value/player_market_value.csv'
        )
        self.player_latest_market_value = pd.read_csv(
            f'{self.data_dir}/player_latest_market_value/player_latest_market_value.csv'
        )
        self.player_performances = pd.read_csv(
            f'{self.data_dir}/player_performances/player_performances.csv'
        )
        self.player_injuries = pd.read_csv(
            f'{self.data_dir}/player_injuries/player_injuries.csv'
        )
        self.transfer_history = pd.read_csv(
            f'{self.data_dir}/transfer_history/transfer_history.csv'
        )
        self.team_details = pd.read_csv(
            f'{self.data_dir}/team_details/team_details.csv'
        )
        self.player_national_performances = pd.read_csv(
            f'{self.data_dir}/player_national_performances/player_national_performances.csv'
        )
        
        print(f"Loaded {len(self.player_profiles)} player profiles")
        print(f"Loaded {len(self.player_market_value)} market value records")
        
        return self
    
    def engineer_features(self):
        print("\nEngineering features...")
        
        df = self.player_profiles.copy()
        
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
        current_date = pd.Timestamp('2024-01-01')
        df['age'] = (current_date - df['date_of_birth']).dt.days / 365.25
        
        df['joined'] = pd.to_datetime(df['joined'], errors='coerce')
        df['years_at_club'] = (current_date - df['joined']).dt.days / 365.25
        
        df['contract_expires'] = pd.to_datetime(df['contract_expires'], errors='coerce')
        df['contract_length'] = (df['contract_expires'] - current_date).dt.days / 365.25
        
        latest_values = self.player_latest_market_value.copy()
        latest_values.columns = ['player_id', 'latest_value_date', 'market_value_eur']
        df = df.merge(latest_values[['player_id', 'market_value_eur']], on='player_id', how='left')
        
        perf_agg = self.player_performances.groupby('player_id').agg({
            'goals': 'sum',
            'assists': 'sum',
            'minutes_played': 'sum',
            'yellow_cards': 'sum',
            'direct_red_cards': 'sum',
            'nb_on_pitch': 'sum'
        }).reset_index()
        perf_agg.columns = ['player_id', 'total_goals', 'total_assists', 'total_minutes', 
                            'total_yellow_cards', 'total_red_cards', 'total_appearances']
        
        perf_agg['goals_per_appearance'] = perf_agg['total_goals'] / (perf_agg['total_appearances'] + 1)
        perf_agg['assists_per_appearance'] = perf_agg['total_assists'] / (perf_agg['total_appearances'] + 1)
        
        df = df.merge(perf_agg, on='player_id', how='left')
        
        injury_agg = self.player_injuries.groupby('player_id').agg({
            'days_missed': 'sum',
            'games_missed': 'sum',
            'injury_reason': 'count'
        }).reset_index()
        injury_agg.columns = ['player_id', 'total_injury_days', 'total_games_missed', 'injury_count']
        df = df.merge(injury_agg, on='player_id', how='left')
        
        transfer_agg = self.transfer_history.groupby('player_id').agg({
            'transfer_fee': ['sum', 'max', 'count']
        }).reset_index()
        transfer_agg.columns = ['player_id', 'total_transfer_fees', 'max_transfer_fee', 'transfer_count']
        df = df.merge(transfer_agg, on='player_id', how='left')
        
        national_agg = self.player_national_performances.groupby('player_id').agg({
            'matches': 'sum',
            'goals': 'sum'
        }).reset_index()
        national_agg.columns = ['player_id', 'national_matches', 'national_goals']
        df = df.merge(national_agg, on='player_id', how='left')
        
        team_agg = self.player_profiles.groupby('current_club_id').agg({
            'player_id': 'count'
        }).reset_index()
        team_agg.columns = ['current_club_id', 'team_squad_size']
        df = df.merge(team_agg, on='current_club_id', how='left')
        
        self.engineered_df = df
        print(f"Engineered features for {len(df)} players")
        
        return self
    
    def preprocess(self):
        print("\nPreprocessing data...")
        
        df = self.engineered_df.copy()
        
        df = df[df['market_value_eur'].notna()]
        df = df[df['market_value_eur'] > 0]
        
        print(f"After filtering valid market values: {len(df)} players")
        
        numeric_features = [
            'age', 'height', 'years_at_club', 'contract_length',
            'total_goals', 'total_assists', 'total_minutes', 
            'total_yellow_cards', 'total_red_cards', 'total_appearances',
            'goals_per_appearance', 'assists_per_appearance',
            'total_injury_days', 'total_games_missed', 'injury_count',
            'total_transfer_fees', 'max_transfer_fee', 'transfer_count',
            'national_matches', 'national_goals', 'team_squad_size'
        ]
        
        categorical_features = ['position', 'main_position', 'foot', 'citizenship']
        
        for col in numeric_features:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)
        
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        categorical_encoded = [f'{col}_encoded' for col in categorical_features if col in df.columns]
        
        self.feature_columns = numeric_features + categorical_encoded
        self.target_column = 'market_value_eur'
        
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        self.processed_df = df
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Features: {self.feature_columns}")
        print(f"Target distribution: min={y.min():.0f}, max={y.max():.0f}, mean={y.mean():.0f}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        print(f"\nSplitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def save_encoders(self, filepath='./models/label_encoders.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.label_encoders, filepath)
        print(f"Label encoders saved to {filepath}")
    
    def run_pipeline(self, test_size=0.2, random_state=42):
        self.load_data()
        self.engineer_features()
        X, y = self.preprocess()
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)
        self.save_encoders()
        
        return X_train, X_test, y_train, y_test, self.feature_columns

if __name__ == "__main__":
    pipeline = FootballDataPipeline(data_dir='../data')
    X_train, X_test, y_train, y_test, feature_cols = pipeline.run_pipeline()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {len(feature_cols)}")