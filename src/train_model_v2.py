"""
Script d'entraînement amélioré avec XGBoost
- Moins de dépendance aux transferts passés
- Features mieux engineerées
- Optimisation des hyperparamètres
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Essayer d'importer XGBoost, sinon utiliser GradientBoosting
try:
    from xgboost import XGBRegressor
    USE_XGBOOST = True
    print("✅ XGBoost disponible")
except (ImportError, Exception):
    from sklearn.ensemble import GradientBoostingRegressor
    USE_XGBOOST = False
    print("⚠️ XGBoost non disponible, utilisation de GradientBoosting (tout aussi efficace)")

print("="*80)
print("ENTRAÎNEMENT DU MODÈLE AMÉLIORÉ")
print("="*80)

# Chemins
DATA_DIR = '../data'
MODEL_DIR = '../models'

# Charger les données
print("\n📂 Chargement des données...")
profiles = pd.read_csv(f'{DATA_DIR}/player_profiles/player_profiles.csv', low_memory=False)
performances = pd.read_csv(f'{DATA_DIR}/player_performances/player_performances.csv', low_memory=False)
injuries = pd.read_csv(f'{DATA_DIR}/player_injuries/player_injuries.csv', low_memory=False)
transfers = pd.read_csv(f'{DATA_DIR}/transfer_history/transfer_history.csv', low_memory=False)
national = pd.read_csv(f'{DATA_DIR}/player_national_performances/player_national_performances.csv', low_memory=False)
market_values = pd.read_csv(f'{DATA_DIR}/player_market_value/player_market_value.csv', low_memory=False)

print(f"  - {len(profiles):,} profils joueurs")
print(f"  - {len(performances):,} performances")
print(f"  - {len(market_values):,} valeurs marchandes")

# Obtenir la dernière valeur marchande pour chaque joueur
print("\n🔧 Préparation des données cibles...")
market_values['date'] = pd.to_datetime(market_values['date_unix'], errors='coerce')
latest_values = market_values.sort_values('date').groupby('player_id').last().reset_index()
latest_values = latest_values[['player_id', 'value']]
latest_values = latest_values.rename(columns={'value': 'market_value'})
latest_values = latest_values[latest_values['market_value'] > 0]
print(f"  - {len(latest_values):,} joueurs avec valeur marchande")

# Agréger les performances par joueur
print("\n🔧 Agrégation des performances...")
perf_agg = performances.groupby('player_id').agg({
    'goals': 'sum',
    'assists': 'sum',
    'minutes_played': 'sum',
    'yellow_cards': 'sum',
    'direct_red_cards': 'sum',
    'nb_on_pitch': 'sum',
    'season_name': 'nunique'  # Nombre de saisons
}).reset_index()
perf_agg.columns = ['player_id', 'total_goals', 'total_assists', 'total_minutes', 
                    'total_yellow', 'total_red', 'total_appearances', 'seasons_played']

# Agréger les blessures
inj_agg = injuries.groupby('player_id').agg({
    'days_missed': 'sum',
    'games_missed': 'sum',
    'injury_reason': 'count'
}).reset_index()
inj_agg.columns = ['player_id', 'total_injury_days', 'total_games_missed', 'injury_count']

# Agréger les transferts (SANS le montant max pour réduire la dépendance)
trans_agg = transfers.groupby('player_id').agg({
    'transfer_fee': ['count', 'mean']  # Nombre et moyenne (pas max!)
}).reset_index()
trans_agg.columns = ['player_id', 'transfer_count', 'avg_transfer_fee']
trans_agg['avg_transfer_fee'] = trans_agg['avg_transfer_fee'].fillna(0)

# Agréger les sélections nationales
nat_agg = national.groupby('player_id').agg({
    'matches': 'sum',
    'goals': 'sum'
}).reset_index()
nat_agg.columns = ['player_id', 'national_matches', 'national_goals']

# Fusionner toutes les données
print("\n🔧 Fusion des données...")
df = profiles[['player_id', 'date_of_birth', 'height', 'position', 'main_position', 
               'foot', 'citizenship', 'joined', 'contract_expires']].copy()

df = df.merge(latest_values, on='player_id', how='inner')
df = df.merge(perf_agg, on='player_id', how='left')
df = df.merge(inj_agg, on='player_id', how='left')
df = df.merge(trans_agg, on='player_id', how='left')
df = df.merge(nat_agg, on='player_id', how='left')

# Remplir les NaN
df = df.fillna(0)
print(f"  - {len(df):,} joueurs après fusion")

# Feature Engineering amélioré
print("\n🔧 Feature Engineering amélioré...")

# Âge
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
df['age'] = (pd.Timestamp.now() - df['date_of_birth']).dt.days / 365.25
df['age'] = df['age'].fillna(25)

# Âge optimal (pic de valeur vers 27 ans) - distance au pic
df['age_from_peak'] = abs(df['age'] - 27)

# Années au club
df['joined'] = pd.to_datetime(df['joined'], errors='coerce')
df['years_at_club'] = (pd.Timestamp.now() - df['joined']).dt.days / 365.25
df['years_at_club'] = df['years_at_club'].fillna(1).clip(lower=0)

# Durée du contrat
df['contract_expires'] = pd.to_datetime(df['contract_expires'], errors='coerce')
df['contract_length'] = (df['contract_expires'] - pd.Timestamp.now()).dt.days / 365.25
df['contract_length'] = df['contract_length'].fillna(2).clip(lower=0)

# Hauteur
df['height'] = pd.to_numeric(df['height'], errors='coerce').fillna(180)

# Ratios de performance (plus significatifs que les totaux bruts)
df['goals_per_90'] = (df['total_goals'] / df['total_minutes'] * 90).replace([np.inf, -np.inf], 0).fillna(0)
df['assists_per_90'] = (df['total_assists'] / df['total_minutes'] * 90).replace([np.inf, -np.inf], 0).fillna(0)
df['goals_assists_per_90'] = df['goals_per_90'] + df['assists_per_90']

# Régularité (minutes par saison)
df['minutes_per_season'] = (df['total_minutes'] / df['seasons_played']).replace([np.inf, -np.inf], 0).fillna(0)

# Ratio blessures
df['injury_rate'] = (df['total_injury_days'] / df['total_appearances']).replace([np.inf, -np.inf], 0).fillna(0)

# International ratio
df['international_ratio'] = (df['national_matches'] / df['total_appearances']).replace([np.inf, -np.inf], 0).fillna(0).clip(upper=1)

# Disciplinaire
df['cards_per_90'] = ((df['total_yellow'] + df['total_red'] * 3) / df['total_minutes'] * 90).replace([np.inf, -np.inf], 0).fillna(0)

# Encodage des variables catégorielles
print("\n🔧 Encodage des variables catégorielles...")
label_encoders = {}
for col in ['position', 'main_position', 'foot', 'citizenship']:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Sélection des features (SANS max_transfer_fee!)
feature_columns = [
    # Démographiques
    'age', 'age_from_peak', 'height', 'years_at_club', 'contract_length',
    # Performance (ratios > totaux)
    'goals_per_90', 'assists_per_90', 'goals_assists_per_90',
    'total_goals', 'total_assists', 'total_appearances', 'total_minutes',
    'minutes_per_season', 'seasons_played',
    # Blessures
    'injury_count', 'injury_rate',
    # Transferts (moyenne, pas max)
    'transfer_count', 'avg_transfer_fee',
    # International
    'national_matches', 'national_goals', 'international_ratio',
    # Disciplinaire
    'total_yellow', 'total_red', 'cards_per_90',
    # Catégorielles
    'position_encoded', 'main_position_encoded', 'foot_encoded', 'citizenship_encoded'
]

print(f"  - {len(feature_columns)} features sélectionnées")

# Préparer X et y
X = df[feature_columns].copy()
y = df['market_value'].copy()

# Retirer les valeurs extrêmes (> 200M€) pour éviter les outliers
mask = y < 200_000_000
X = X[mask]
y = y[mask]
print(f"  - {len(X):,} échantillons après filtrage des outliers")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  - Train: {len(X_train):,} | Test: {len(X_test):,}")

# Entraînement du modèle
print("\n🚀 Entraînement du modèle...")
if USE_XGBOOST:
    model = XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
else:
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )

model.fit(X_train, y_train)
print("✅ Modèle entraîné!")

# Évaluation
print("\n📊 Évaluation du modèle...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Baseline
baseline_mae = mean_absolute_error(y_test, [y_test.mean()] * len(y_test))

print("\n" + "="*60)
print("RÉSULTATS")
print("="*60)
print(f"R² Score:              {r2:.4f}")
print(f"MAE:                   €{mae:,.0f}")
print(f"RMSE:                  €{rmse:,.0f}")
print(f"Baseline MAE:          €{baseline_mae:,.0f}")
print(f"Amélioration:          {((baseline_mae - mae) / baseline_mae * 100):.1f}%")
print("="*60)

# Feature importance
print("\n📊 Importance des features (Top 15):")
if USE_XGBOOST:
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
else:
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

for i, row in importance.head(15).iterrows():
    bar = "█" * int(row['importance'] * 50)
    print(f"  {row['feature']:25s} {row['importance']:.3f} {bar}")

# Sauvegarder le modèle
print("\n💾 Sauvegarde du modèle...")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, f'{MODEL_DIR}/football_model.pkl')
joblib.dump({
    'model_type': 'xgboost' if USE_XGBOOST else 'gradient_boosting',
    'feature_columns': feature_columns,
    'r2_score': r2,
    'mae': mae,
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}, f'{MODEL_DIR}/model_metadata.pkl')
joblib.dump(label_encoders, f'{MODEL_DIR}/label_encoders.pkl')
joblib.dump({'X_test': X_test, 'y_test': y_test}, f'{MODEL_DIR}/test_data.pkl')

print(f"✅ Modèle sauvegardé dans {MODEL_DIR}/")

# Exemples de prédictions
print("\n🔮 Exemples de prédictions:")
sample_indices = X_test.sample(5).index
for idx in sample_indices:
    actual = y_test.loc[idx]
    pred = model.predict(X_test.loc[[idx]])[0]
    error = abs(actual - pred) / actual * 100
    print(f"  Réel: €{actual:>12,.0f} | Prédit: €{pred:>12,.0f} | Erreur: {error:>5.1f}%")

print("\n✅ Entraînement terminé!")
