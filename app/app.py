"""
Football Player Market Value Predictor - Version 2.0
API Flask moderne avec comparateur, simulations et explications IA
"""
from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
import numpy as np
import os
import io
import json
import warnings
import unicodedata
from datetime import datetime
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'football_model.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'models', 'model_metadata.pkl')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================
print("🚀 Chargement du modèle...")
model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)
feature_columns = metadata['feature_columns']
print(f"✅ Modèle chargé avec {len(feature_columns)} features")

print("📊 Chargement des données...")
profiles = pd.read_csv(os.path.join(DATA_DIR, 'player_profiles', 'player_profiles.csv'), low_memory=False)
performances = pd.read_csv(os.path.join(DATA_DIR, 'player_performances', 'player_performances.csv'), low_memory=False)
injuries = pd.read_csv(os.path.join(DATA_DIR, 'player_injuries', 'player_injuries.csv'), low_memory=False)
transfers = pd.read_csv(os.path.join(DATA_DIR, 'transfer_history', 'transfer_history.csv'), low_memory=False)
national = pd.read_csv(os.path.join(DATA_DIR, 'player_national_performances', 'player_national_performances.csv'), low_memory=False)
market_values = pd.read_csv(os.path.join(DATA_DIR, 'player_market_value', 'player_market_value.csv'), low_memory=False)
print(f"✅ {len(profiles)} joueurs chargés")

# ============================================================================
# CONSTANTES
# ============================================================================
TOP_CLUBS = [
    'paris saint-germain', 'psg', 'real madrid', 'fc barcelona', 'barcelona',
    'manchester city', 'manchester united', 'liverpool', 'chelsea', 'arsenal',
    'bayern munich', 'bayern münchen', 'borussia dortmund', 'juventus',
    'inter milan', 'ac milan', 'napoli', 'atletico madrid', 'atlético madrid',
    'tottenham', 'newcastle', 'aston villa', 'benfica', 'porto', 'sporting',
    'ajax', 'rb leipzig', 'bayer leverkusen', 'monaco', 'marseille', 'lyon',
    'lille', 'al-hilal', 'al-nassr', 'al-ittihad', 'al-ahli'
]

POSITION_CATEGORIES = {
    'Goalkeeper': ['Goalkeeper'],
    'Defender': ['Centre-Back', 'Left-Back', 'Right-Back', 'Defender'],
    'Midfielder': ['Central Midfield', 'Defensive Midfield', 'Attacking Midfield', 'Left Midfield', 'Right Midfield'],
    'Forward': ['Centre-Forward', 'Left Winger', 'Right Winger', 'Second Striker', 'Attack']
}

# Importance relative des features (calculée à partir du modèle)
FEATURE_IMPORTANCE = {
    'age': 0.15, 'age_from_peak': 0.12, 'goals_per_90': 0.18, 'assists_per_90': 0.14,
    'total_goals': 0.08, 'total_assists': 0.06, 'total_appearances': 0.05,
    'national_matches': 0.10, 'national_goals': 0.04, 'avg_transfer_fee': 0.03,
    'injury_rate': 0.02, 'contract_length': 0.02, 'height': 0.01
}

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================
def normalize_text(text):
    """Retire les accents d'un texte"""
    if pd.isna(text):
        return ''
    return unicodedata.normalize('NFD', str(text)).encode('ascii', 'ignore').decode('utf-8').lower()

def format_currency(value):
    """Formate une valeur en euros"""
    if value >= 1_000_000:
        return f"€{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"€{value/1_000:.0f}K"
    return f"€{value:.0f}"

def get_player_score(row):
    """Score de pertinence pour trier les résultats"""
    score = 0
    club = str(row.get('current_club_name', '')).lower()
    for top_club in TOP_CLUBS:
        if top_club in club:
            score += 100
            break
    if 'retired' in club: score -= 50
    if 'unknown' in club: score -= 50
    if any(x in club for x in ['u19', 'u21', 'sub-', ' ii', ' b']): score -= 30
    return score

def get_position_category(position):
    """Retourne la catégorie de position"""
    if pd.isna(position):
        return 'Unknown'
    position = str(position)
    for category, positions in POSITION_CATEGORIES.items():
        if any(p.lower() in position.lower() for p in positions):
            return category
    return 'Unknown'

# ============================================================================
# PRÉPARATION DES FEATURES
# ============================================================================
def prepare_player_features(player_id, age_override=None, goals_override=None, assists_override=None):
    """Prépare les features pour un joueur (avec possibilité de simulation)"""
    
    player_df = profiles[profiles['player_id'] == player_id]
    if len(player_df) == 0:
        raise ValueError(f"Joueur {player_id} non trouvé")
    player = player_df.iloc[0]
    
    # Âge (avec override pour simulation)
    age = 25
    if age_override is not None:
        age = age_override
    elif pd.notna(player.get('date_of_birth')):
        try:
            dob = pd.to_datetime(player['date_of_birth'])
            age = (pd.Timestamp.now() - dob).days / 365.25
        except:
            pass
    
    age_from_peak = abs(age - 27)
    
    # Hauteur
    height = player.get('height', 180)
    if pd.isna(height) or height == 0:
        height = 180
    
    # Années au club
    years_at_club = 1
    if pd.notna(player.get('joined')):
        try:
            joined = pd.to_datetime(player['joined'])
            years_at_club = max(0, (pd.Timestamp.now() - joined).days / 365.25)
        except:
            pass
    
    # Contrat
    contract_length = 2
    if pd.notna(player.get('contract_expires')):
        try:
            expires = pd.to_datetime(player['contract_expires'])
            contract_length = max(0, (expires - pd.Timestamp.now()).days / 365.25)
        except:
            pass
    
    # Performances - CALCUL CORRIGÉ
    player_perf = performances[performances['player_id'] == player_id]
    total_goals = int(player_perf['goals'].sum()) if 'goals' in player_perf.columns else 0
    total_assists = int(player_perf['assists'].sum()) if 'assists' in player_perf.columns else 0
    total_minutes = int(player_perf['minutes_played'].sum()) if 'minutes_played' in player_perf.columns else 0
    total_yellow = int(player_perf['yellow_cards'].sum()) if 'yellow_cards' in player_perf.columns else 0
    total_red = int(player_perf['direct_red_cards'].sum()) if 'direct_red_cards' in player_perf.columns else 0
    total_appearances = int(player_perf['nb_on_pitch'].sum()) if 'nb_on_pitch' in player_perf.columns else len(player_perf)
    seasons_played = player_perf['season_name'].nunique() if 'season_name' in player_perf.columns else 1
    
    # Override pour simulation
    if goals_override is not None:
        total_goals = goals_override
    if assists_override is not None:
        total_assists = assists_override
    
    # STATS RÉALISTES - buts/assists par MATCH (pas par 90min avec minutes fausses)
    # Un bon buteur fait 0.5-0.8 buts/match, un excellent fait 0.8-1.0
    goals_per_match = (total_goals / total_appearances) if total_appearances > 0 else 0
    assists_per_match = (total_assists / total_appearances) if total_appearances > 0 else 0
    
    # Pour le modèle, on garde goals_per_90 mais calculé correctement
    # En moyenne un joueur joue ~75-80 min par match
    avg_minutes_per_match = (total_minutes / total_appearances) if total_appearances > 0 else 75
    goals_per_90 = (goals_per_match / avg_minutes_per_match * 90) if avg_minutes_per_match > 0 else 0
    assists_per_90 = (assists_per_match / avg_minutes_per_match * 90) if avg_minutes_per_match > 0 else 0
    goals_assists_per_90 = goals_per_90 + assists_per_90
    
    minutes_per_season = (total_minutes / seasons_played) if seasons_played > 0 else 0
    
    # Blessures
    player_inj = injuries[injuries['player_id'] == player_id]
    total_injury_days = int(player_inj['days_missed'].sum()) if 'days_missed' in player_inj.columns else 0
    injury_count = len(player_inj)
    injury_rate = (total_injury_days / total_appearances) if total_appearances > 0 else 0
    
    # Transferts
    player_trans = transfers[transfers['player_id'] == player_id]
    avg_transfer_fee = 0
    if 'transfer_fee' in player_trans.columns:
        fees = pd.to_numeric(player_trans['transfer_fee'], errors='coerce').fillna(0)
        avg_transfer_fee = float(fees.mean()) if len(fees) > 0 else 0
    transfer_count = len(player_trans)
    
    # Équipe nationale
    player_nat = national[national['player_id'] == player_id]
    national_matches = int(player_nat['matches'].sum()) if 'matches' in player_nat.columns else 0
    national_goals = int(player_nat['goals'].sum()) if 'goals' in player_nat.columns else 0
    international_ratio = min(1, national_matches / total_appearances) if total_appearances > 0 else 0
    
    cards_per_90 = ((total_yellow + total_red * 3) / total_minutes * 90) if total_minutes > 0 else 0
    
    # Encodages
    position_encoded = hash(str(player.get('position', ''))) % 1000
    main_position_encoded = hash(str(player.get('main_position', ''))) % 1000
    foot_encoded = hash(str(player.get('foot', ''))) % 100
    citizenship_encoded = hash(str(player.get('citizenship', ''))) % 1000
    
    features = {
        'age': float(age),
        'age_from_peak': float(age_from_peak),
        'height': float(height),
        'years_at_club': float(years_at_club),
        'contract_length': float(contract_length),
        'goals_per_90': float(goals_per_90),
        'assists_per_90': float(assists_per_90),
        'goals_assists_per_90': float(goals_assists_per_90),
        'total_goals': int(total_goals),
        'total_assists': int(total_assists),
        'total_appearances': int(total_appearances),
        'total_minutes': int(total_minutes),
        'minutes_per_season': float(minutes_per_season),
        'seasons_played': int(seasons_played),
        'injury_count': int(injury_count),
        'injury_rate': float(injury_rate),
        'transfer_count': int(transfer_count),
        'avg_transfer_fee': float(avg_transfer_fee),
        'national_matches': int(national_matches),
        'national_goals': int(national_goals),
        'international_ratio': float(international_ratio),
        'total_yellow': int(total_yellow),
        'total_red': int(total_red),
        'cards_per_90': float(cards_per_90),
        'position_encoded': int(position_encoded),
        'main_position_encoded': int(main_position_encoded),
        'foot_encoded': int(foot_encoded),
        'citizenship_encoded': int(citizenship_encoded),
        # Stats additionnelles pour l'affichage (pas pour le modèle)
        'goals_per_match': float(goals_per_match),
        'assists_per_match': float(assists_per_match),
        'avg_minutes_per_match': float(avg_minutes_per_match),
        'national_goals_ratio': float(national_goals / national_matches) if national_matches > 0 else 0,
    }
    
    return features

def predict_value(features):
    """Prédit la valeur à partir des features"""
    feature_vector = [features.get(col, 0) for col in feature_columns]
    return float(model.predict([feature_vector])[0])

def get_peak_age(position):
    """Retourne l'âge de pic selon la position"""
    position = str(position).lower() if position else ''
    if 'goalkeeper' in position or 'keeper' in position:
        return 29  # Gardiens maturent plus tard
    elif 'defence' in position or 'back' in position or 'defender' in position:
        return 28  # Défenseurs pic un peu plus tard
    elif 'midfield' in position:
        return 27  # Milieux
    else:  # Attack, Forward, etc.
        return 26  # Attaquants pic plus tôt (vitesse/explosivité)

def predict_future_value(player_id, features, current_value, position):
    """
    Prédit la valeur future d'un joueur en simulant sa progression jusqu'à son pic.
    Retourne une projection réaliste basée sur les tendances football.
    """
    age = features.get('age', 25)
    peak_age = get_peak_age(position)
    
    # Stats actuelles
    total_goals = features.get('total_goals', 0)
    total_assists = features.get('total_assists', 0)
    total_appearances = features.get('total_appearances', 0)
    national_matches = features.get('national_matches', 0)
    national_goals = features.get('national_goals', 0)
    
    # Calculer les ratios actuels (performance par saison)
    career_years = max(1, age - 18)  # Années de carrière pro
    goals_per_year = total_goals / career_years
    assists_per_year = total_assists / career_years
    apps_per_year = total_appearances / career_years
    national_per_year = national_matches / career_years
    
    projections = []
    
    # Projeter pour chaque année future jusqu'à 34 ans
    for future_age in range(int(age) + 1, 35):
        years_ahead = future_age - int(age)
        
        # Facteur de progression selon l'âge par rapport au pic
        if future_age <= peak_age:
            # Avant le pic: progression (jeunes progressent plus vite)
            if age < 21:
                progression_factor = 1.15  # Pépites: +15% par an
            elif age < 24:
                progression_factor = 1.10  # Jeunes talents: +10% par an
            else:
                progression_factor = 1.05  # Joueurs confirmés: +5% par an
        else:
            # Après le pic: déclin progressif
            years_past_peak = future_age - peak_age
            if years_past_peak <= 2:
                progression_factor = 0.95  # -5% par an juste après pic
            elif years_past_peak <= 4:
                progression_factor = 0.88  # -12% par an
            else:
                progression_factor = 0.75  # -25% par an après 32+
        
        # Calculer le facteur cumulé
        if future_age <= peak_age:
            cumulative_factor = progression_factor ** years_ahead
        else:
            # Avant pic puis déclin
            years_to_peak = max(0, peak_age - int(age))
            years_after_peak = future_age - peak_age
            
            # Croissance jusqu'au pic
            if age < 21:
                growth_factor = 1.15 ** years_to_peak
            elif age < 24:
                growth_factor = 1.10 ** years_to_peak
            else:
                growth_factor = 1.05 ** years_to_peak
            
            # Déclin après pic
            decline_factor = 1.0
            for y in range(1, years_after_peak + 1):
                if y <= 2:
                    decline_factor *= 0.95
                elif y <= 4:
                    decline_factor *= 0.88
                else:
                    decline_factor *= 0.75
            
            cumulative_factor = growth_factor * decline_factor
        
        # Projeter les stats futures
        projected_goals = total_goals + (goals_per_year * years_ahead * (1 + (cumulative_factor - 1) * 0.3))
        projected_assists = total_assists + (assists_per_year * years_ahead * (1 + (cumulative_factor - 1) * 0.3))
        projected_apps = total_appearances + (apps_per_year * years_ahead * 0.95)  # Légère baisse des matchs avec l'âge
        projected_national = national_matches + (national_per_year * years_ahead * (1 if future_age < 32 else 0.5))
        projected_national_goals = national_goals + (national_goals / max(1, national_matches) * (projected_national - national_matches))
        
        # Créer les features futures
        future_features = features.copy()
        future_features['age'] = future_age
        future_features['total_goals'] = projected_goals
        future_features['total_assists'] = projected_assists
        future_features['total_appearances'] = projected_apps
        future_features['national_matches'] = projected_national
        future_features['national_goals'] = projected_national_goals
        future_features['goals_per_match'] = projected_goals / max(1, projected_apps)
        future_features['assists_per_match'] = projected_assists / max(1, projected_apps)
        
        # Prédire la valeur future
        future_value = predict_value(future_features)
        
        # Appliquer un facteur de correction réaliste pour les jeunes talents
        if age < 23 and future_age <= peak_age:
            # Les jeunes avec bon ratio ont plus de potentiel
            current_goal_ratio = total_goals / max(1, total_appearances)
            if current_goal_ratio > 0.3:  # Très bon finisseur
                future_value *= 1.15
            elif current_goal_ratio > 0.2:
                future_value *= 1.08
        
        projections.append({
            'age': future_age,
            'year': 2026 + years_ahead,
            'value': max(0, future_value),
            'is_peak': future_age == peak_age
        })
    
    # Trouver la valeur pic
    peak_projection = max(projections, key=lambda x: x['value']) if projections else None
    
    return {
        'current_age': int(age),
        'peak_age': peak_age,
        'current_value': current_value,
        'peak_value': peak_projection['value'] if peak_projection else current_value,
        'peak_year': peak_projection['year'] if peak_projection else 2026,
        'projections': projections,
        'potential_increase': ((peak_projection['value'] / current_value - 1) * 100) if peak_projection and current_value > 0 else 0
    }

def get_feature_explanation(features, predicted_value):
    """Génère une explication des facteurs influençant la prédiction - VERSION FOOTBALL EXPERT"""
    explanations = []
    
    # ===== ÂGE ET POTENTIEL =====
    age = features.get('age', 25)
    if age < 21:
        explanations.append({
            'factor': 'Pépite',
            'impact': 'very_positive',
            'icon': '💎',
            'text': f"À {age:.0f} ans, énorme potentiel de revente et progression",
            'weight': 0.18
        })
    elif age < 24:
        explanations.append({
            'factor': 'Jeune talent',
            'impact': 'positive',
            'icon': '🌟',
            'text': f"À {age:.0f} ans, encore une belle marge de progression",
            'weight': 0.15
        })
    elif age <= 29:
        explanations.append({
            'factor': 'Prime de carrière',
            'impact': 'positive',
            'icon': '💪',
            'text': f"À {age:.0f} ans, dans ses meilleures années",
            'weight': 0.10
        })
    elif age <= 32:
        explanations.append({
            'factor': 'Expérience',
            'impact': 'neutral',
            'icon': '📊',
            'text': f"À {age:.0f} ans, valeur marchande en déclin mais expérience précieuse",
            'weight': 0.08
        })
    else:
        explanations.append({
            'factor': 'Fin de carrière',
            'impact': 'negative',
            'icon': '📉',
            'text': f"À {age:.0f} ans, faible valeur de revente",
            'weight': 0.12
        })
    
    # ===== EFFICACITÉ DEVANT LE BUT =====
    goals_per_match = features.get('goals_per_match', 0)
    total_goals = features.get('total_goals', 0)
    total_appearances = features.get('total_appearances', 0)
    
    if goals_per_match >= 0.8:
        explanations.append({
            'factor': 'Buteur d\'élite',
            'impact': 'very_positive',
            'icon': '⚽',
            'text': f"{total_goals} buts en {total_appearances} matchs ({goals_per_match:.2f}/match) - Niveau Ballon d'Or",
            'weight': 0.22
        })
    elif goals_per_match >= 0.5:
        explanations.append({
            'factor': 'Très bon buteur',
            'impact': 'very_positive',
            'icon': '⚽',
            'text': f"{total_goals} buts en {total_appearances} matchs ({goals_per_match:.2f}/match) - Top niveau européen",
            'weight': 0.18
        })
    elif goals_per_match >= 0.3:
        explanations.append({
            'factor': 'Buteur régulier',
            'impact': 'positive',
            'icon': '⚽',
            'text': f"{total_goals} buts en {total_appearances} matchs ({goals_per_match:.2f}/match) - Bon rendement",
            'weight': 0.12
        })
    elif goals_per_match >= 0.15 and total_goals > 10:
        explanations.append({
            'factor': 'Contribution offensive',
            'impact': 'neutral',
            'icon': '⚽',
            'text': f"{total_goals} buts en {total_appearances} matchs - Apport offensif correct",
            'weight': 0.06
        })
    
    # ===== PASSES DÉCISIVES =====
    assists_per_match = features.get('assists_per_match', 0)
    total_assists = features.get('total_assists', 0)
    
    if assists_per_match >= 0.5:
        explanations.append({
            'factor': 'Passeur exceptionnel',
            'impact': 'very_positive',
            'icon': '🎯',
            'text': f"{total_assists} passes D. en {total_appearances} matchs ({assists_per_match:.2f}/match) - Vision de jeu exceptionnelle",
            'weight': 0.18
        })
    elif assists_per_match >= 0.3:
        explanations.append({
            'factor': 'Excellent passeur',
            'impact': 'positive',
            'icon': '🎯',
            'text': f"{total_assists} passes D. en {total_appearances} matchs ({assists_per_match:.2f}/match) - Créateur de haut niveau",
            'weight': 0.14
        })
    elif assists_per_match >= 0.15 and total_assists > 10:
        explanations.append({
            'factor': 'Bon créateur',
            'impact': 'positive',
            'icon': '🎯',
            'text': f"{total_assists} passes D. en {total_appearances} matchs - Capable de faire la différence",
            'weight': 0.08
        })
    
    # ===== SÉLECTIONS NATIONALES =====
    national_matches = features.get('national_matches', 0)
    national_goals = features.get('national_goals', 0)
    national_goals_ratio = features.get('national_goals_ratio', 0)
    
    if national_matches >= 100:
        explanations.append({
            'factor': 'Légende internationale',
            'impact': 'very_positive',
            'icon': '🏆',
            'text': f"{national_matches} sélections, {national_goals} buts - Statut de légende",
            'weight': 0.15
        })
    elif national_matches >= 50:
        explanations.append({
            'factor': 'International confirmé',
            'impact': 'very_positive',
            'icon': '🌍',
            'text': f"{national_matches} sélections, {national_goals} buts - Cadre de sa sélection",
            'weight': 0.12
        })
    elif national_matches >= 20:
        explanations.append({
            'factor': 'International',
            'impact': 'positive',
            'icon': '🌍',
            'text': f"{national_matches} sélections - Reconnu au niveau international",
            'weight': 0.08
        })
    elif national_matches >= 5:
        explanations.append({
            'factor': 'Sélectionné',
            'impact': 'positive',
            'icon': '🏳️',
            'text': f"{national_matches} sélections - A goûté à l'équipe nationale",
            'weight': 0.04
        })
    
    # ===== BLESSURES ET ROBUSTESSE =====
    injury_count = features.get('injury_count', 0)
    total_injury_days = features.get('injury_rate', 0) * total_appearances if total_appearances > 0 else 0
    seasons_played = features.get('seasons_played', 1)
    
    injuries_per_season = injury_count / seasons_played if seasons_played > 0 else 0
    
    if injury_count == 0 or injuries_per_season < 0.5:
        explanations.append({
            'factor': 'Robustesse',
            'impact': 'positive',
            'icon': '💚',
            'text': f"Rarement blessé - Grande fiabilité physique",
            'weight': 0.08
        })
    elif injuries_per_season > 2:
        explanations.append({
            'factor': 'Fragilité',
            'impact': 'negative',
            'icon': '🏥',
            'text': f"{injury_count} blessures sur {seasons_played} saisons - Risque médical",
            'weight': 0.10
        })
    elif injuries_per_season > 1:
        explanations.append({
            'factor': 'Blessures',
            'impact': 'negative',
            'icon': '🤕',
            'text': f"Historique de blessures à surveiller",
            'weight': 0.06
        })
    
    # ===== CONTRAT =====
    contract_length = features.get('contract_length', 2)
    if contract_length < 1:
        explanations.append({
            'factor': 'Fin de contrat',
            'impact': 'negative',
            'icon': '📝',
            'text': f"Moins d'un an de contrat - Risque de départ libre",
            'weight': 0.10
        })
    elif contract_length < 2:
        explanations.append({
            'factor': 'Contrat court',
            'impact': 'negative',
            'icon': '📝',
            'text': f"{contract_length:.1f} ans restants - Position de négociation faible pour le club",
            'weight': 0.06
        })
    elif contract_length >= 4:
        explanations.append({
            'factor': 'Contrat long',
            'impact': 'positive',
            'icon': '✍️',
            'text': f"{contract_length:.1f} ans de contrat - Le club contrôle la situation",
            'weight': 0.05
        })
    
    # ===== EXPÉRIENCE =====
    if total_appearances >= 400:
        explanations.append({
            'factor': 'Vétéran',
            'impact': 'positive',
            'icon': '🎖️',
            'text': f"{total_appearances} matchs pro - Énorme expérience du haut niveau",
            'weight': 0.06
        })
    elif total_appearances >= 200:
        explanations.append({
            'factor': 'Expérimenté',
            'impact': 'positive',
            'icon': '📈',
            'text': f"{total_appearances} matchs pro - Joueur confirmé",
            'weight': 0.04
        })
    
    # Trier par poids et retourner les 6 plus importants
    explanations.sort(key=lambda x: x['weight'], reverse=True)
    return explanations[:6]

def get_player_history(player_id):
    """Récupère l'historique des valeurs marchandes"""
    player_mv = market_values[market_values['player_id'] == player_id].copy()
    if len(player_mv) == 0:
        return []
    
    # Convertir les dates - la colonne 'date_unix' contient en fait des dates au format string YYYY-MM-DD
    date_col = 'date_unix' if 'date_unix' in player_mv.columns else 'date'
    player_mv['date'] = pd.to_datetime(player_mv[date_col], errors='coerce')
    
    player_mv = player_mv.dropna(subset=['date'])
    player_mv = player_mv.sort_values('date')
    
    history = []
    for _, row in player_mv.iterrows():
        # La colonne s'appelle 'value' dans le CSV
        value = row.get('value', row.get('market_value', 0))
        if pd.notna(value) and value > 0:
            history.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'value': float(value)
            })
    
    return history[-20:]  # Dernières 20 valeurs max

# ============================================================================
# RECHERCHE DE JOUEURS
# ============================================================================
def search_player(name, limit=15):
    """Recherche intelligente de joueurs"""
    name_normalized = normalize_text(name)
    mask = profiles['player_name'].apply(lambda x: name_normalized in normalize_text(x))
    matches = profiles[mask].copy()
    
    if len(matches) == 0:
        return []
    
    matches['_score'] = matches.apply(get_player_score, axis=1)
    matches = matches.sort_values('_score', ascending=False)
    
    result = matches[['player_id', 'player_name', 'position', 'current_club_name', 'date_of_birth', 'citizenship']].head(limit).copy()
    result = result.rename(columns={'player_name': 'name'})
    result['name'] = result['name'].str.replace(r'\s*\(\d+\)$', '', regex=True)
    
    # Calculer l'âge
    result['age'] = result['date_of_birth'].apply(lambda x: 
        int((pd.Timestamp.now() - pd.to_datetime(x)).days / 365.25) if pd.notna(x) else None
    )
    
    result = result.fillna('')
    return result.to_dict('records')

def get_top_players(position_category=None, limit=20):
    """Retourne les joueurs les mieux valorisés par position"""
    # Joindre avec les dernières valeurs marchandes
    latest_mv = market_values.sort_values('date_unix' if 'date_unix' in market_values.columns else 'date', ascending=False)
    latest_mv = latest_mv.drop_duplicates(subset=['player_id'], keep='first')
    
    # La colonne s'appelle 'value' dans le CSV
    value_col = 'value' if 'value' in latest_mv.columns else 'market_value'
    latest_mv = latest_mv.rename(columns={value_col: 'market_value'})
    
    merged = profiles.merge(latest_mv[['player_id', 'market_value']], on='player_id', how='inner')
    merged = merged[merged['market_value'] > 0]
    
    if position_category and position_category != 'All':
        positions = POSITION_CATEGORIES.get(position_category, [])
        if positions:
            mask = merged['position'].apply(lambda x: any(p.lower() in str(x).lower() for p in positions))
            merged = merged[mask]
    
    merged = merged.sort_values('market_value', ascending=False).head(limit)
    
    result = []
    for _, row in merged.iterrows():
        player_name = str(row['player_name'])
        # Retirer l'ID entre parenthèses du nom
        import re
        player_name = re.sub(r'\s*\(\d+\)$', '', player_name)
        result.append({
            'player_id': int(row['player_id']),
            'name': player_name,
            'position': row.get('position', ''),
            'club': row.get('current_club_name', ''),
            'market_value': float(row['market_value']),
            'citizenship': row.get('citizenship', '')
        })
    
    return result

# ============================================================================
# TEMPLATE HTML MODERNE
# ============================================================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚽ Football Market Value AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        :root {
            --bg-primary: #0f0f23;
            --bg-secondary: #1a1a35;
            --bg-card: #252545;
            --text-primary: #ffffff;
            --text-secondary: #a0a0b0;
            --accent: #00d4ff;
            --accent-secondary: #7b2cbf;
            --success: #00c853;
            --warning: #ffc107;
            --danger: #ff5252;
            --gradient: linear-gradient(135deg, var(--accent) 0%, var(--accent-secondary) 100%);
        }
        
        [data-theme="light"] {
            --bg-primary: #f5f5f7;
            --bg-secondary: #ffffff;
            --bg-card: #ffffff;
            --text-primary: #1a1a2e;
            --text-secondary: #666680;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            transition: all 0.3s ease;
        }
        
        .app-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            margin-bottom: 30px;
        }
        
        .logo {
            font-size: 1.8em;
            font-weight: 700;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .header-actions {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .theme-toggle {
            background: var(--bg-card);
            border: none;
            padding: 10px 15px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1.2em;
            transition: all 0.2s;
        }
        
        .theme-toggle:hover {
            transform: scale(1.1);
        }
        
        /* Tabs */
        .tabs {
            display: flex;
            gap: 5px;
            background: var(--bg-secondary);
            padding: 5px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        
        .tab {
            flex: 1;
            padding: 15px 20px;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            font-size: 1em;
            font-weight: 500;
            cursor: pointer;
            border-radius: 12px;
            transition: all 0.2s;
        }
        
        .tab:hover {
            color: var(--text-primary);
        }
        
        .tab.active {
            background: var(--gradient);
            color: white;
        }
        
        .tab-content {
            display: none;
            animation: fadeIn 0.3s ease;
        }
        
        .tab-content.active {
            display: block;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Search Box */
        .search-section {
            background: var(--bg-secondary);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .search-container {
            position: relative;
        }
        
        .search-input {
            width: 100%;
            padding: 18px 25px;
            font-size: 1.1em;
            border: 2px solid transparent;
            border-radius: 15px;
            background: var(--bg-card);
            color: var(--text-primary);
            transition: all 0.2s;
        }
        
        .search-input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
        }
        
        .search-input::placeholder {
            color: var(--text-secondary);
        }
        
        .autocomplete-list {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: var(--bg-card);
            border-radius: 0 0 15px 15px;
            max-height: 400px;
            overflow-y: auto;
            z-index: 1000;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            display: none;
        }
        
        .autocomplete-item {
            padding: 15px 25px;
            cursor: pointer;
            border-bottom: 1px solid var(--bg-secondary);
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.15s;
        }
        
        .autocomplete-item:hover, .autocomplete-item.selected {
            background: rgba(0, 212, 255, 0.1);
        }
        
        .player-info-mini {
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        
        .player-name-mini {
            font-weight: 600;
            font-size: 1.05em;
        }
        
        .player-meta-mini {
            font-size: 0.85em;
            color: var(--text-secondary);
        }
        
        .player-club-badge {
            background: var(--bg-secondary);
            padding: 5px 12px;
            border-radius: 8px;
            font-size: 0.8em;
            color: var(--text-secondary);
        }
        
        /* Cards */
        .card {
            background: var(--bg-secondary);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 20px;
        }
        
        .card-title {
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* Player Result Card */
        .player-result {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        @media (max-width: 900px) {
            .player-result {
                grid-template-columns: 1fr;
            }
        }
        
        .player-header {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .player-avatar {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: var(--gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2em;
        }
        
        .player-main-info h2 {
            font-size: 1.6em;
            margin-bottom: 5px;
        }
        
        .player-main-info p {
            color: var(--text-secondary);
        }
        
        .value-display {
            background: var(--gradient);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            margin-bottom: 25px;
        }
        
        .value-label {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }
        
        .value-amount {
            font-size: 2.8em;
            font-weight: 700;
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }
        
        .stat-box {
            background: var(--bg-card);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5em;
            font-weight: 700;
            color: var(--accent);
        }
        
        .stat-label {
            font-size: 0.8em;
            color: var(--text-secondary);
            margin-top: 5px;
        }
        
        /* Explanation */
        .explanation-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .explanation-item {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 15px;
            background: var(--bg-card);
            border-radius: 12px;
            border-left: 4px solid var(--accent);
        }
        
        .explanation-item.positive { border-left-color: var(--success); }
        .explanation-item.very_positive { border-left-color: #00ff88; }
        .explanation-item.negative { border-left-color: var(--danger); }
        
        .explanation-icon {
            font-size: 1.5em;
        }
        
        .explanation-text {
            flex: 1;
        }
        
        .explanation-factor {
            font-weight: 600;
            margin-bottom: 3px;
        }
        
        .explanation-detail {
            font-size: 0.9em;
            color: var(--text-secondary);
        }
        
        /* Comparison */
        .compare-section {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 20px;
            align-items: start;
        }
        
        @media (max-width: 900px) {
            .compare-section {
                grid-template-columns: 1fr;
            }
        }
        
        .vs-badge {
            background: var(--gradient);
            padding: 15px 25px;
            border-radius: 50px;
            font-weight: 700;
            font-size: 1.2em;
            align-self: center;
        }
        
        .compare-player {
            background: var(--bg-card);
            border-radius: 15px;
            padding: 20px;
        }
        
        .compare-value {
            font-size: 1.8em;
            font-weight: 700;
            color: var(--accent);
            margin: 15px 0;
        }
        
        /* Radar Chart Container */
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        
        /* Simulation */
        .simulation-controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .control-group label {
            font-weight: 500;
            color: var(--text-secondary);
        }
        
        .control-group input[type="range"] {
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: var(--bg-card);
            appearance: none;
        }
        
        .control-group input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--accent);
            cursor: pointer;
        }
        
        .control-value {
            text-align: center;
            font-size: 1.2em;
            font-weight: 600;
            color: var(--accent);
        }
        
        .simulation-result {
            background: var(--bg-card);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
        }
        
        .simulation-comparison {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 30px;
            margin-top: 20px;
        }
        
        .sim-value {
            text-align: center;
        }
        
        .sim-label {
            font-size: 0.9em;
            color: var(--text-secondary);
            margin-bottom: 5px;
        }
        
        .sim-amount {
            font-size: 1.8em;
            font-weight: 700;
        }
        
        .sim-amount.current { color: var(--text-secondary); }
        .sim-amount.simulated { color: var(--accent); }
        
        .sim-arrow {
            font-size: 2em;
            color: var(--text-secondary);
        }
        
        .sim-diff {
            margin-top: 15px;
            padding: 10px 20px;
            border-radius: 10px;
            font-weight: 600;
        }
        
        .sim-diff.positive { background: rgba(0, 200, 83, 0.2); color: var(--success); }
        .sim-diff.negative { background: rgba(255, 82, 82, 0.2); color: var(--danger); }
        
        /* Future Projection Card */
        .future-projection-card {
            background: linear-gradient(145deg, rgba(0, 212, 255, 0.1) 0%, rgba(123, 44, 191, 0.1) 100%);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 16px;
            padding: 25px;
            margin-top: 25px;
        }
        
        .future-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 20px;
        }
        
        .future-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .future-stats {
            display: flex;
            gap: 30px;
        }
        
        .future-stat {
            text-align: center;
        }
        
        .future-stat-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 5px;
        }
        
        .future-stat-value {
            font-size: 1.3rem;
            font-weight: 700;
        }
        
        .future-stat-value.current { color: var(--text-secondary); }
        .future-stat-value.peak { color: var(--success); }
        .future-stat-value.increase { color: var(--warning); }
        
        .future-chart {
            height: 280px;
            margin-top: 15px;
        }
        
        .peak-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 12px;
            background: linear-gradient(135deg, var(--warning) 0%, #ff9800 100%);
            color: #000;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        /* Top Players */
        .filter-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .filter-btn {
            padding: 10px 20px;
            border: none;
            background: var(--bg-card);
            color: var(--text-secondary);
            border-radius: 10px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .filter-btn:hover {
            background: var(--bg-secondary);
            color: var(--text-primary);
        }
        
        .filter-btn.active {
            background: var(--gradient);
            color: white;
        }
        
        .top-players-grid {
            display: grid;
            gap: 10px;
        }
        
        .top-player-row {
            display: flex;
            align-items: center;
            padding: 15px 20px;
            background: var(--bg-card);
            border-radius: 12px;
            gap: 20px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .top-player-row:hover {
            transform: translateX(5px);
            background: rgba(0, 212, 255, 0.1);
        }
        
        .rank {
            font-size: 1.2em;
            font-weight: 700;
            width: 40px;
            color: var(--accent);
        }
        
        .top-player-info {
            flex: 1;
        }
        
        .top-player-name {
            font-weight: 600;
        }
        
        .top-player-club {
            font-size: 0.85em;
            color: var(--text-secondary);
        }
        
        .top-player-value {
            font-weight: 700;
            color: var(--success);
        }
        
        /* History Chart - Modern Style */
        .history-chart-container {
            background: linear-gradient(145deg, var(--bg-card) 0%, rgba(37, 37, 69, 0.7) 100%);
            border-radius: 16px;
            padding: 25px;
            margin-top: 25px;
            border: 1px solid rgba(0, 212, 255, 0.1);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        }
        
        .history-chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .history-chart-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .history-chart-stats {
            display: flex;
            gap: 20px;
        }
        
        .history-stat {
            text-align: center;
            padding: 8px 15px;
            background: rgba(0, 212, 255, 0.1);
            border-radius: 10px;
        }
        
        .history-stat-value {
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--accent);
        }
        
        .history-stat-label {
            font-size: 0.7rem;
            color: var(--text-secondary);
            margin-top: 2px;
        }
        
        .history-chart {
            height: 320px;
            position: relative;
        }
        
        /* Export Buttons */
        .export-buttons {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        .export-btn {
            padding: 12px 25px;
            border: none;
            background: var(--bg-card);
            color: var(--text-primary);
            border-radius: 10px;
            cursor: pointer;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
        }
        
        .export-btn:hover {
            background: var(--accent);
            color: white;
        }
        
        /* Loading */
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 50px;
            gap: 15px;
        }
        
        .spinner {
            width: 30px;
            height: 30px;
            border: 3px solid var(--bg-card);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Responsive */
        @media (max-width: 600px) {
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .tabs {
                flex-wrap: wrap;
            }
            
            .tab {
                flex: 1 1 45%;
            }
            
            .value-amount {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Header -->
        <header class="header">
            <div class="logo">
                <span>⚽</span>
                <span>Football Market Value AI</span>
            </div>
            <div class="header-actions">
                <button class="theme-toggle" onclick="toggleTheme()" title="Changer le thème">🌙</button>
            </div>
        </header>
        
        <!-- Tabs -->
        <div class="tabs">
            <button class="tab active" onclick="showTab('search')">🔍 Recherche</button>
            <button class="tab" onclick="showTab('compare')">⚖️ Comparer</button>
            <button class="tab" onclick="showTab('simulate')">🎮 Simulation</button>
            <button class="tab" onclick="showTab('top')">🏆 Top Joueurs</button>
        </div>
        
        <!-- Search Tab -->
        <div id="tab-search" class="tab-content active">
            <div class="search-section">
                <div class="search-container">
                    <input type="text" class="search-input" id="searchInput" 
                           placeholder="Rechercher un joueur (ex: Mbappé, Haaland, Bellingham...)" autocomplete="off">
                    <div id="autocomplete" class="autocomplete-list"></div>
                </div>
            </div>
            <div id="searchResult"></div>
        </div>
        
        <!-- Compare Tab -->
        <div id="tab-compare" class="tab-content">
            <div class="card">
                <h3 class="card-title">⚖️ Comparer deux joueurs</h3>
                <div class="compare-section">
                    <div>
                        <div class="search-container">
                            <input type="text" class="search-input" id="compareInput1" 
                                   placeholder="Premier joueur..." autocomplete="off">
                            <div id="autocomplete1" class="autocomplete-list"></div>
                        </div>
                        <div id="comparePlayer1" class="compare-player" style="margin-top: 15px; display: none;"></div>
                    </div>
                    <div class="vs-badge">VS</div>
                    <div>
                        <div class="search-container">
                            <input type="text" class="search-input" id="compareInput2" 
                                   placeholder="Deuxième joueur..." autocomplete="off">
                            <div id="autocomplete2" class="autocomplete-list"></div>
                        </div>
                        <div id="comparePlayer2" class="compare-player" style="margin-top: 15px; display: none;"></div>
                    </div>
                </div>
                <div id="radarChartContainer" style="display: none;">
                    <div class="chart-container">
                        <canvas id="radarChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Simulate Tab -->
        <div id="tab-simulate" class="tab-content">
            <div class="card">
                <h3 class="card-title">🎮 Simulation What-If</h3>
                <div class="search-container" style="margin-bottom: 25px;">
                    <input type="text" class="search-input" id="simulateInput" 
                           placeholder="Sélectionner un joueur à simuler..." autocomplete="off">
                    <div id="autocompleteSimulate" class="autocomplete-list"></div>
                </div>
                <div id="simulationPanel" style="display: none;">
                    <div class="simulation-controls">
                        <div class="control-group">
                            <label>🎂 Âge</label>
                            <input type="range" id="simAge" min="16" max="40" value="25">
                            <div class="control-value" id="simAgeValue">25 ans</div>
                        </div>
                        <div class="control-group">
                            <label>⚽ Buts totaux</label>
                            <input type="range" id="simGoals" min="0" max="500" value="50">
                            <div class="control-value" id="simGoalsValue">50 buts</div>
                        </div>
                        <div class="control-group">
                            <label>🎯 Passes décisives</label>
                            <input type="range" id="simAssists" min="0" max="300" value="30">
                            <div class="control-value" id="simAssistsValue">30 assists</div>
                        </div>
                    </div>
                    <div id="simulationResult" class="simulation-result"></div>
                </div>
            </div>
        </div>
        
        <!-- Top Players Tab -->
        <div id="tab-top" class="tab-content">
            <div class="card">
                <h3 class="card-title">🏆 Top Joueurs par Valeur Marchande</h3>
                <div class="filter-bar">
                    <button class="filter-btn active" onclick="filterTopPlayers('All')">Tous</button>
                    <button class="filter-btn" onclick="filterTopPlayers('Forward')">Attaquants</button>
                    <button class="filter-btn" onclick="filterTopPlayers('Midfielder')">Milieux</button>
                    <button class="filter-btn" onclick="filterTopPlayers('Defender')">Défenseurs</button>
                    <button class="filter-btn" onclick="filterTopPlayers('Goalkeeper')">Gardiens</button>
                </div>
                <div id="topPlayersGrid" class="top-players-grid"></div>
            </div>
        </div>
    </div>

    <script>
        // ============================================================================
        // STATE
        // ============================================================================
        let currentPlayer = null;
        let comparePlayer1 = null;
        let comparePlayer2 = null;
        let simulatePlayer = null;
        let radarChart = null;
        let historyChart = null;
        
        // ============================================================================
        // THEME
        // ============================================================================
        function toggleTheme() {
            const body = document.body;
            const btn = document.querySelector('.theme-toggle');
            if (body.getAttribute('data-theme') === 'light') {
                body.removeAttribute('data-theme');
                btn.textContent = '🌙';
            } else {
                body.setAttribute('data-theme', 'light');
                btn.textContent = '☀️';
            }
        }
        
        // ============================================================================
        // TABS
        // ============================================================================
        function showTab(tabId) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelector(`[onclick="showTab('${tabId}')"]`).classList.add('active');
            document.getElementById(`tab-${tabId}`).classList.add('active');
            
            if (tabId === 'top') {
                loadTopPlayers('All');
            }
        }
        
        // ============================================================================
        // SEARCH AUTOCOMPLETE
        // ============================================================================
        function setupAutocomplete(inputId, autocompleteId, onSelect) {
            const input = document.getElementById(inputId);
            const autocomplete = document.getElementById(autocompleteId);
            
            if (!input || !autocomplete) {
                console.error('Autocomplete elements not found:', inputId, autocompleteId);
                return;
            }
            
            let selectedIndex = -1;
            let currentResults = [];
            let debounceTimer = null;
            
            input.addEventListener('input', function() {
                if (debounceTimer) clearTimeout(debounceTimer);
                const name = this.value.trim();
                
                if (name.length < 2) {
                    autocomplete.style.display = 'none';
                    return;
                }
                
                debounceTimer = setTimeout(async () => {
                    try {
                        const response = await fetch('/api/search?name=' + encodeURIComponent(name));
                        const data = await response.json();
                        currentResults = data.players || [];
                        selectedIndex = -1;
                        
                        if (currentResults.length > 0) {
                            autocomplete.innerHTML = currentResults.map((p, i) => `
                                <div class="autocomplete-item" data-index="${i}">
                                    <div class="player-info-mini">
                                        <div class="player-name-mini">${p.name}</div>
                                        <div class="player-meta-mini">${p.position || ''} ${p.age ? '• ' + p.age + ' ans' : ''}</div>
                                    </div>
                                    <span class="player-club-badge">${p.current_club_name || 'N/A'}</span>
                                </div>
                            `).join('');
                            
                            autocomplete.querySelectorAll('.autocomplete-item').forEach((item, i) => {
                                item.addEventListener('click', () => {
                                    input.value = currentResults[i].name;
                                    autocomplete.style.display = 'none';
                                    onSelect(currentResults[i]);
                                });
                            });
                            
                            autocomplete.style.display = 'block';
                        } else {
                            autocomplete.innerHTML = '<div style="padding: 15px; color: var(--text-secondary);">Aucun joueur trouvé</div>';
                            autocomplete.style.display = 'block';
                        }
                    } catch (error) {
                        console.error('Erreur de recherche:', error);
                    }
                }, 250);
            });
            
            input.addEventListener('keydown', function(e) {
                const items = autocomplete.querySelectorAll('.autocomplete-item');
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    selectedIndex = Math.min(selectedIndex + 1, items.length - 1);
                    items.forEach((item, i) => item.classList.toggle('selected', i === selectedIndex));
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    selectedIndex = Math.max(selectedIndex - 1, 0);
                    items.forEach((item, i) => item.classList.toggle('selected', i === selectedIndex));
                } else if (e.key === 'Enter' && selectedIndex >= 0) {
                    e.preventDefault();
                    input.value = currentResults[selectedIndex].name;
                    autocomplete.style.display = 'none';
                    onSelect(currentResults[selectedIndex]);
                } else if (e.key === 'Escape') {
                    autocomplete.style.display = 'none';
                }
            });
            
            document.addEventListener('click', (e) => {
                if (!e.target.closest('.search-container')) {
                    autocomplete.style.display = 'none';
                }
            });
        }
        
        // ============================================================================
        // MAIN SEARCH
        // ============================================================================
        async function selectPlayer(player) {
            currentPlayer = player;
            const resultDiv = document.getElementById('searchResult');
            resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><span>Analyse en cours...</span></div>';
            
            try {
                const response = await fetch('/api/predict?player_id=' + player.player_id);
                const data = await response.json();
                
                if (data.error) {
                    resultDiv.innerHTML = `<div class="card" style="color: var(--danger);">❌ ${data.error}</div>`;
                    return;
                }
                
                const value = formatCurrency(data.predicted_value);
                const explanations = data.explanations || [];
                const history = data.history || [];
                const future = data.future || null;
                
                let html = `
                    <div class="player-result">
                        <div class="card">
                            <div class="player-header">
                                <div class="player-avatar">⚽</div>
                                <div class="player-main-info">
                                    <h2>${player.name}</h2>
                                    <p>${player.position || ''} • ${player.current_club_name || 'N/A'}</p>
                                </div>
                            </div>
                            <div class="value-display">
                                <div class="value-label">💰 Valeur Marchande Estimée</div>
                                <div class="value-amount">${value}</div>
                                ${future && future.potential_increase > 10 ? `
                                    <div class="peak-badge">🚀 Potentiel: ${formatCurrency(future.peak_value)} à ${future.peak_age} ans</div>
                                ` : ''}
                                <div class="value-label">💰 Valeur Marchande Estimée</div>
                                <div class="value-amount">${value}</div>
                            </div>
                            <div class="stats-grid">
                                <div class="stat-box">
                                    <div class="stat-value">${Math.round(data.features.age || 0)}</div>
                                    <div class="stat-label">Âge</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-value">${data.features.total_goals || 0}</div>
                                    <div class="stat-label">Buts</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-value">${data.features.total_assists || 0}</div>
                                    <div class="stat-label">Passes D.</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-value">${data.features.total_appearances || 0}</div>
                                    <div class="stat-label">Matchs</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-value">${data.features.national_matches || 0}</div>
                                    <div class="stat-label">Sélections</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-value">${(data.features.goals_per_match || 0).toFixed(2)}</div>
                                    <div class="stat-label">Buts/Match</div>
                                </div>
                            </div>
                            ${history.length > 0 ? `
                                <div class="history-chart-container">
                                    <div class="history-chart-header">
                                        <div class="history-chart-title">
                                            📈 Évolution de la valeur marchande
                                        </div>
                                        <div class="history-chart-stats">
                                            <div class="history-stat">
                                                <div class="history-stat-value">${formatCurrency(Math.max(...history.map(h => h.value)))}</div>
                                                <div class="history-stat-label">Maximum</div>
                                            </div>
                                            <div class="history-stat">
                                                <div class="history-stat-value">${formatCurrency(history[history.length-1].value)}</div>
                                                <div class="history-stat-label">Actuelle</div>
                                            </div>
                                            <div class="history-stat">
                                                <div class="history-stat-value">${history.length}</div>
                                                <div class="history-stat-label">Points</div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="history-chart">
                                        <canvas id="historyChart"></canvas>
                                    </div>
                                </div>
                            ` : ''}
                            ${future && future.projections && future.projections.length > 0 ? `
                                <div class="future-projection-card">
                                    <div class="future-header">
                                        <div class="future-title">
                                            🔮 Projection de carrière
                                        </div>
                                        <div class="future-stats">
                                            <div class="future-stat">
                                                <div class="future-stat-label">Valeur actuelle</div>
                                                <div class="future-stat-value current">${formatCurrency(future.current_value)}</div>
                                            </div>
                                            <div class="future-stat">
                                                <div class="future-stat-label">Pic estimé (${future.peak_age} ans)</div>
                                                <div class="future-stat-value peak">${formatCurrency(future.peak_value)}</div>
                                            </div>
                                            <div class="future-stat">
                                                <div class="future-stat-label">Potentiel</div>
                                                <div class="future-stat-value increase">${future.potential_increase > 0 ? '+' : ''}${future.potential_increase.toFixed(0)}%</div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="future-chart">
                                        <canvas id="futureChart"></canvas>
                                    </div>
                                </div>
                            ` : ''}
                            <div class="export-buttons">
                                <button class="export-btn" onclick="exportCSV()">📊 Export CSV</button>
                                <button class="export-btn" onclick="exportJSON()">📋 Export JSON</button>
                            </div>
                        </div>
                        <div class="card">
                            <h3 class="card-title">🧠 Pourquoi cette valeur ?</h3>
                            <div class="explanation-list">
                                ${explanations.map(e => `
                                    <div class="explanation-item ${e.impact}">
                                        <span class="explanation-icon">${e.icon}</span>
                                        <div class="explanation-text">
                                            <div class="explanation-factor">${e.factor}</div>
                                            <div class="explanation-detail">${e.text}</div>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                `;
                
                resultDiv.innerHTML = html;
                
                // Dessiner le graphique historique avec dates réelles
                if (history.length > 0) {
                    const ctx = document.getElementById('historyChart').getContext('2d');
                    if (historyChart) historyChart.destroy();
                    
                    // Créer un gradient moderne
                    const gradient = ctx.createLinearGradient(0, 0, 0, 320);
                    gradient.addColorStop(0, 'rgba(0, 212, 255, 0.5)');
                    gradient.addColorStop(0.3, 'rgba(0, 212, 255, 0.25)');
                    gradient.addColorStop(0.7, 'rgba(123, 44, 191, 0.1)');
                    gradient.addColorStop(1, 'rgba(123, 44, 191, 0)');
                    
                    // Convertir les dates
                    const chartData = history.map(h => ({
                        x: new Date(h.date),
                        y: h.value
                    }));
                    
                    // Calculer les extremes
                    const values = history.map(h => h.value);
                    const maxValue = Math.max(...values);
                    const maxIndex = values.indexOf(maxValue);
                    
                    // Déterminer l'unité temporelle automatiquement
                    const firstDate = new Date(history[0].date);
                    const lastDate = new Date(history[history.length - 1].date);
                    const diffYears = (lastDate - firstDate) / (1000 * 60 * 60 * 24 * 365);
                    const timeUnit = diffYears > 3 ? 'year' : diffYears > 1 ? 'quarter' : 'month';
                    
                    historyChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            datasets: [{
                                label: 'Valeur marchande',
                                data: chartData,
                                borderColor: '#00d4ff',
                                backgroundColor: gradient,
                                fill: true,
                                tension: 0.4,
                                borderWidth: 3,
                                pointBackgroundColor: chartData.map((d, i) => 
                                    i === maxIndex ? '#ffc107' : '#00d4ff'
                                ),
                                pointBorderColor: chartData.map((d, i) => 
                                    i === maxIndex ? '#fff' : 'rgba(255,255,255,0.8)'
                                ),
                                pointBorderWidth: chartData.map((d, i) => 
                                    i === maxIndex ? 3 : 2
                                ),
                                pointRadius: chartData.map((d, i) => 
                                    i === maxIndex ? 8 : (i === chartData.length - 1 ? 6 : 3)
                                ),
                                pointHoverRadius: 10,
                                pointHoverBackgroundColor: '#fff',
                                pointHoverBorderColor: '#00d4ff',
                                pointHoverBorderWidth: 3
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            interaction: {
                                intersect: false,
                                mode: 'index'
                            },
                            plugins: {
                                legend: { display: false },
                                tooltip: {
                                    backgroundColor: 'rgba(15, 15, 35, 0.98)',
                                    titleColor: '#00d4ff',
                                    bodyColor: '#fff',
                                    borderColor: 'rgba(0, 212, 255, 0.5)',
                                    borderWidth: 1,
                                    padding: 15,
                                    cornerRadius: 12,
                                    displayColors: false,
                                    titleFont: { size: 14, weight: '600', family: 'Inter' },
                                    bodyFont: { size: 15, weight: '700', family: 'Inter' },
                                    footerFont: { size: 11, family: 'Inter' },
                                    callbacks: {
                                        title: function(items) {
                                            const date = new Date(items[0].parsed.x);
                                            return '📅 ' + date.toLocaleDateString('fr-FR', { 
                                                day: 'numeric', 
                                                month: 'long', 
                                                year: 'numeric' 
                                            });
                                        },
                                        label: function(item) {
                                            return '💰 ' + formatCurrency(item.parsed.y);
                                        },
                                        footer: function(items) {
                                            const idx = items[0].dataIndex;
                                            if (idx === maxIndex) {
                                                return '⭐ Record historique';
                                            }
                                            if (idx > 0) {
                                                const prev = chartData[idx - 1].y;
                                                const curr = items[0].parsed.y;
                                                const diff = ((curr - prev) / prev * 100).toFixed(1);
                                                const emoji = diff > 0 ? '📈' : '📉';
                                                return emoji + ' ' + (diff > 0 ? '+' : '') + diff + '% vs précédent';
                                            }
                                            return '';
                                        }
                                    }
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: false,
                                    grace: '10%',
                                    ticks: {
                                        callback: v => formatCurrency(v),
                                        color: '#a0a0b0',
                                        font: { size: 11, family: 'Inter' },
                                        padding: 12,
                                        maxTicksLimit: 6
                                    },
                                    grid: { 
                                        color: 'rgba(255,255,255,0.05)',
                                        drawBorder: false
                                    },
                                    border: { display: false }
                                },
                                x: {
                                    type: 'time',
                                    time: {
                                        unit: timeUnit,
                                        displayFormats: {
                                            month: 'MMM yyyy',
                                            quarter: 'MMM yyyy',
                                            year: 'yyyy'
                                        }
                                    },
                                    ticks: { 
                                        color: '#a0a0b0',
                                        font: { size: 11, family: 'Inter' },
                                        maxRotation: 0,
                                        autoSkip: true,
                                        maxTicksLimit: 8
                                    },
                                    grid: { display: false },
                                    border: { display: false }
                                }
                            },
                            animation: {
                                duration: 1200,
                                easing: 'easeOutQuart',
                                delay: (context) => context.dataIndex * 50
                            }
                        }
                    });
                }
                
                // Dessiner le graphique de projection future
                if (future && future.projections && future.projections.length > 0) {
                    const futureCtx = document.getElementById('futureChart').getContext('2d');
                    
                    // Gradient pour la zone avant le pic
                    const futureGradient = futureCtx.createLinearGradient(0, 0, 0, 280);
                    futureGradient.addColorStop(0, 'rgba(0, 200, 83, 0.4)');
                    futureGradient.addColorStop(0.5, 'rgba(0, 200, 83, 0.15)');
                    futureGradient.addColorStop(1, 'rgba(0, 200, 83, 0)');
                    
                    // Données du graphique
                    const projectionData = future.projections.map(p => ({
                        x: p.year,
                        y: p.value
                    }));
                    
                    // Point actuel
                    const currentPoint = {
                        x: 2026,
                        y: future.current_value
                    };
                    
                    // Trouver l'index du pic
                    const peakIndex = future.projections.findIndex(p => p.is_peak);
                    
                    new Chart(futureCtx, {
                        type: 'line',
                        data: {
                            datasets: [
                                {
                                    label: 'Valeur actuelle',
                                    data: [currentPoint],
                                    borderColor: '#00d4ff',
                                    backgroundColor: '#00d4ff',
                                    pointRadius: 10,
                                    pointBorderWidth: 3,
                                    pointBorderColor: '#fff',
                                    pointStyle: 'circle',
                                    showLine: false
                                },
                                {
                                    label: 'Projection future',
                                    data: projectionData,
                                    borderColor: '#00c853',
                                    backgroundColor: futureGradient,
                                    fill: true,
                                    tension: 0.4,
                                    borderWidth: 3,
                                    borderDash: [5, 5],
                                    pointBackgroundColor: projectionData.map((_, i) => 
                                        i === peakIndex ? '#ffc107' : '#00c853'
                                    ),
                                    pointBorderColor: projectionData.map((_, i) => 
                                        i === peakIndex ? '#fff' : 'rgba(255,255,255,0.6)'
                                    ),
                                    pointRadius: projectionData.map((_, i) => 
                                        i === peakIndex ? 10 : 3
                                    ),
                                    pointBorderWidth: projectionData.map((_, i) => 
                                        i === peakIndex ? 3 : 1
                                    )
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            interaction: {
                                intersect: false,
                                mode: 'index'
                            },
                            plugins: {
                                legend: { display: false },
                                tooltip: {
                                    backgroundColor: 'rgba(15, 15, 35, 0.98)',
                                    titleColor: '#00c853',
                                    bodyColor: '#fff',
                                    borderColor: 'rgba(0, 200, 83, 0.5)',
                                    borderWidth: 1,
                                    padding: 15,
                                    cornerRadius: 12,
                                    displayColors: false,
                                    callbacks: {
                                        title: function(items) {
                                            const year = items[0].parsed.x;
                                            const proj = future.projections.find(p => p.year === year);
                                            if (proj && proj.is_peak) {
                                                return '⭐ ' + year + ' - Âge de pic (' + proj.age + ' ans)';
                                            }
                                            return '📅 ' + year + (proj ? ' (' + proj.age + ' ans)' : '');
                                        },
                                        label: function(item) {
                                            return '💰 ' + formatCurrency(item.parsed.y);
                                        },
                                        footer: function(items) {
                                            const value = items[0].parsed.y;
                                            const diff = ((value / future.current_value - 1) * 100).toFixed(0);
                                            if (diff > 0) return '📈 +' + diff + '% vs actuel';
                                            if (diff < 0) return '📉 ' + diff + '% vs actuel';
                                            return '';
                                        }
                                    }
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: false,
                                    grace: '10%',
                                    ticks: {
                                        callback: v => formatCurrency(v),
                                        color: '#a0a0b0',
                                        font: { size: 11, family: 'Inter' },
                                        maxTicksLimit: 6
                                    },
                                    grid: { 
                                        color: 'rgba(255,255,255,0.05)',
                                        drawBorder: false
                                    },
                                    border: { display: false }
                                },
                                x: {
                                    type: 'linear',
                                    ticks: { 
                                        color: '#a0a0b0',
                                        font: { size: 11, family: 'Inter' },
                                        stepSize: 1,
                                        callback: v => v.toString()
                                    },
                                    grid: { display: false },
                                    border: { display: false }
                                }
                            },
                            animation: {
                                duration: 1500,
                                easing: 'easeOutQuart'
                            }
                        }
                    });
                }
                
            } catch (error) {
                resultDiv.innerHTML = `<div class="card" style="color: var(--danger);">❌ Erreur: ${error.message}</div>`;
            }
        }
        
        // ============================================================================
        // COMPARISON
        // ============================================================================
        async function selectComparePlayer(player, num) {
            if (num === 1) comparePlayer1 = player;
            else comparePlayer2 = player;
            
            const div = document.getElementById(`comparePlayer${num}`);
            div.style.display = 'block';
            div.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
            
            try {
                const response = await fetch('/api/predict?player_id=' + player.player_id);
                const data = await response.json();
                
                if (num === 1) comparePlayer1.data = data;
                else comparePlayer2.data = data;
                
                div.innerHTML = `
                    <h4>${player.name}</h4>
                    <div class="compare-value">${formatCurrency(data.predicted_value)}</div>
                    <p style="color: var(--text-secondary);">${player.position || ''}</p>
                    <p style="color: var(--text-secondary);">${player.current_club_name || ''}</p>
                `;
                
                // Si les deux joueurs sont sélectionnés, afficher le radar
                if (comparePlayer1?.data && comparePlayer2?.data) {
                    drawRadarChart();
                }
            } catch (error) {
                div.innerHTML = `<p style="color: var(--danger);">Erreur: ${error.message}</p>`;
            }
        }
        
        function drawRadarChart() {
            const container = document.getElementById('radarChartContainer');
            container.style.display = 'block';
            
            const ctx = document.getElementById('radarChart').getContext('2d');
            if (radarChart) radarChart.destroy();
            
            const f1 = comparePlayer1.data.features;
            const f2 = comparePlayer2.data.features;
            
            // Normaliser les valeurs pour le radar (0-100)
            const normalize = (v1, v2, max) => {
                const m = Math.max(v1, v2, max);
                return [v1/m*100, v2/m*100];
            };
            
            const [g1, g2] = normalize(f1.total_goals, f2.total_goals, 100);
            const [a1, a2] = normalize(f1.total_assists, f2.total_assists, 100);
            const [m1, m2] = normalize(f1.total_appearances, f2.total_appearances, 500);
            const [n1, n2] = normalize(f1.national_matches, f2.national_matches, 100);
            const gpm1 = (f1.goals_per_match || 0) * 100;
            const gpm2 = (f2.goals_per_match || 0) * 100;
            const apm1 = (f1.assists_per_match || 0) * 100;
            const apm2 = (f2.assists_per_match || 0) * 100;
            const [gp1, gp2] = normalize(gpm1, gpm2, 100);
            const [ap1, ap2] = normalize(apm1, apm2, 100);
            
            radarChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['⚽ Buts', '🎯 Passes D.', '📊 Matchs', '🏆 Sélections', '💥 Buts/Match', '👟 Assists/Match'],
                    datasets: [
                        {
                            label: comparePlayer1.name,
                            data: [g1, a1, m1, n1, gp1, ap1],
                            borderColor: '#00d4ff',
                            backgroundColor: 'rgba(0, 212, 255, 0.15)',
                            borderWidth: 3,
                            pointBackgroundColor: '#00d4ff',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2,
                            pointRadius: 5,
                            pointHoverRadius: 8
                        },
                        {
                            label: comparePlayer2.name,
                            data: [g2, a2, m2, n2, gp2, ap2],
                            borderColor: '#7b2cbf',
                            backgroundColor: 'rgba(123, 44, 191, 0.15)',
                            borderWidth: 3,
                            pointBackgroundColor: '#7b2cbf',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2,
                            pointRadius: 5,
                            pointHoverRadius: 8
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 100,
                            ticks: { 
                                display: false,
                                stepSize: 20
                            },
                            grid: { 
                                color: 'rgba(255,255,255,0.08)',
                                circular: true
                            },
                            angleLines: {
                                color: 'rgba(255,255,255,0.1)'
                            },
                            pointLabels: { 
                                color: '#e0e0e0', 
                                font: { size: 12, weight: '500', family: 'Inter' }
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { 
                                color: '#a0a0b0',
                                font: { size: 12, family: 'Inter' },
                                padding: 20,
                                usePointStyle: true,
                                pointStyle: 'circle'
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(15, 15, 35, 0.95)',
                            titleColor: '#fff',
                            bodyColor: '#fff',
                            borderColor: 'rgba(0, 212, 255, 0.3)',
                            borderWidth: 1,
                            padding: 12,
                            cornerRadius: 10
                        }
                    },
                    animation: {
                        duration: 1000,
                        easing: 'easeOutQuart'
                    }
                }
            });
        }
        
        // ============================================================================
        // SIMULATION
        // ============================================================================
        async function selectSimulatePlayer(player) {
            simulatePlayer = player;
            const panel = document.getElementById('simulationPanel');
            panel.style.display = 'block';
            
            // Charger les données actuelles
            const response = await fetch('/api/predict?player_id=' + player.player_id);
            const data = await response.json();
            simulatePlayer.data = data;
            
            // Initialiser les sliders
            document.getElementById('simAge').value = Math.round(data.features.age);
            document.getElementById('simGoals').value = data.features.total_goals;
            document.getElementById('simAssists').value = data.features.total_assists;
            
            updateSimulation();
        }
        
        async function updateSimulation() {
            if (!simulatePlayer) return;
            
            const age = parseInt(document.getElementById('simAge').value);
            const goals = parseInt(document.getElementById('simGoals').value);
            const assists = parseInt(document.getElementById('simAssists').value);
            
            document.getElementById('simAgeValue').textContent = age + ' ans';
            document.getElementById('simGoalsValue').textContent = goals + ' buts';
            document.getElementById('simAssistsValue').textContent = assists + ' assists';
            
            const response = await fetch(`/api/simulate?player_id=${simulatePlayer.player_id}&age=${age}&goals=${goals}&assists=${assists}`);
            const data = await response.json();
            
            const currentValue = simulatePlayer.data.predicted_value;
            const newValue = data.predicted_value;
            const diff = newValue - currentValue;
            const diffPercent = ((diff / currentValue) * 100).toFixed(1);
            
            document.getElementById('simulationResult').innerHTML = `
                <h4 style="margin-bottom: 15px;">📊 Résultat de la simulation pour ${simulatePlayer.name}</h4>
                <div class="simulation-comparison">
                    <div class="sim-value">
                        <div class="sim-label">Valeur actuelle</div>
                        <div class="sim-amount current">${formatCurrency(currentValue)}</div>
                    </div>
                    <div class="sim-arrow">→</div>
                    <div class="sim-value">
                        <div class="sim-label">Valeur simulée</div>
                        <div class="sim-amount simulated">${formatCurrency(newValue)}</div>
                    </div>
                </div>
                <div class="sim-diff ${diff >= 0 ? 'positive' : 'negative'}">
                    ${diff >= 0 ? '📈' : '📉'} ${diff >= 0 ? '+' : ''}${formatCurrency(diff)} (${diff >= 0 ? '+' : ''}${diffPercent}%)
                </div>
            `;
        }
        
        // ============================================================================
        // TOP PLAYERS
        // ============================================================================
        async function loadTopPlayers(position) {
            const grid = document.getElementById('topPlayersGrid');
            grid.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
            
            try {
                const response = await fetch('/api/top-players?position=' + position);
                const data = await response.json();
                
                grid.innerHTML = data.players.map((p, i) => `
                    <div class="top-player-row" data-id="${p.player_id}" data-name="${p.name.replace(/"/g, '&quot;')}">
                        <div class="rank">#${i + 1}</div>
                        <div class="top-player-info">
                            <div class="top-player-name">${p.name}</div>
                            <div class="top-player-club">${p.club || 'N/A'} • ${p.position || ''}</div>
                        </div>
                        <div class="top-player-value">${formatCurrency(p.market_value)}</div>
                    </div>
                `).join('');
                
                // Ajouter les event listeners
                grid.querySelectorAll('.top-player-row').forEach(row => {
                    row.addEventListener('click', () => {
                        selectTopPlayer(parseInt(row.dataset.id), row.dataset.name);
                    });
                });
            } catch (error) {
                grid.innerHTML = `<p style="color: var(--danger);">Erreur: ${error.message}</p>`;
            }
        }
        
        function filterTopPlayers(position) {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            loadTopPlayers(position);
        }
        
        function selectTopPlayer(id, name) {
            showTab('search');
            document.getElementById('searchInput').value = name;
            selectPlayer({ player_id: id, name: name });
        }
        
        // ============================================================================
        // EXPORT
        // ============================================================================
        function exportCSV() {
            if (!currentPlayer?.data) return;
            const d = currentPlayer.data;
            const f = d.features;
            
            let csv = 'Attribut,Valeur\\n';
            csv += `Joueur,${currentPlayer.name}\\n`;
            csv += `Valeur Prédite,${d.predicted_value}\\n`;
            csv += `Âge,${f.age}\\n`;
            csv += `Buts,${f.total_goals}\\n`;
            csv += `Passes D.,${f.total_assists}\\n`;
            csv += `Matchs,${f.total_appearances}\\n`;
            csv += `Sélections,${f.national_matches}\\n`;
            csv += `Buts/90min,${f.goals_per_90}\\n`;
            
            downloadFile(csv, `${currentPlayer.name.replace(/\\s/g, '_')}_valuation.csv`, 'text/csv');
        }
        
        function exportJSON() {
            if (!currentPlayer?.data) return;
            const json = JSON.stringify({
                player: currentPlayer.name,
                player_id: currentPlayer.player_id,
                predicted_value: currentPlayer.data.predicted_value,
                features: currentPlayer.data.features,
                explanations: currentPlayer.data.explanations
            }, null, 2);
            
            downloadFile(json, `${currentPlayer.name.replace(/\\s/g, '_')}_valuation.json`, 'application/json');
        }
        
        function downloadFile(content, filename, type) {
            const blob = new Blob([content], { type });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }
        
        // ============================================================================
        // UTILS
        // ============================================================================
        function formatCurrency(value) {
            if (value >= 1000000) return '€' + (value / 1000000).toFixed(1) + 'M';
            if (value >= 1000) return '€' + (value / 1000).toFixed(0) + 'K';
            return '€' + value.toFixed(0);
        }
        
        // ============================================================================
        // INIT
        // ============================================================================
        document.addEventListener('DOMContentLoaded', function() {
            try {
                setupAutocomplete('searchInput', 'autocomplete', selectPlayer);
                setupAutocomplete('compareInput1', 'autocomplete1', p => selectComparePlayer(p, 1));
                setupAutocomplete('compareInput2', 'autocomplete2', p => selectComparePlayer(p, 2));
                setupAutocomplete('simulateInput', 'autocompleteSimulate', selectSimulatePlayer);
                
                // Simulation sliders
                ['simAge', 'simGoals', 'simAssists'].forEach(id => {
                    const el = document.getElementById(id);
                    if (el) el.addEventListener('input', updateSimulation);
                });
                
                // Charger les top joueurs au démarrage
                loadTopPlayers('All');
                
                console.log('✅ Application initialisée');
            } catch (error) {
                console.error('❌ Erreur initialisation:', error);
            }
        });
    </script>
</body>
</html>
'''

# ============================================================================
# ROUTES API
# ============================================================================
@app.route('/')
def home():
    return HTML_TEMPLATE

@app.route('/api/search')
def api_search():
    name = request.args.get('name', '')
    if not name:
        return jsonify({'error': 'Nom requis', 'players': []})
    players = search_player(name)
    return jsonify({'players': players})

@app.route('/api/predict')
def api_predict():
    player_id = request.args.get('player_id')
    if not player_id:
        return jsonify({'error': 'player_id requis'})
    
    try:
        player_id = int(player_id)
        if player_id not in profiles['player_id'].values:
            return jsonify({'error': 'Joueur non trouvé'})
        
        # Récupérer la position du joueur
        player_profile = profiles[profiles['player_id'] == player_id].iloc[0]
        position = player_profile.get('position', '')
        
        features = prepare_player_features(player_id)
        predicted_value = predict_value(features)
        explanations = get_feature_explanation(features, predicted_value)
        history = get_player_history(player_id)
        
        # Prédiction de valeur future
        future_prediction = predict_future_value(player_id, features, predicted_value, position)
        
        return jsonify({
            'player_id': player_id,
            'predicted_value': predicted_value,
            'features': features,
            'explanations': explanations,
            'history': history,
            'future': future_prediction
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/api/simulate')
def api_simulate():
    player_id = request.args.get('player_id')
    age = request.args.get('age', type=int)
    goals = request.args.get('goals', type=int)
    assists = request.args.get('assists', type=int)
    
    if not player_id:
        return jsonify({'error': 'player_id requis'})
    
    try:
        player_id = int(player_id)
        features = prepare_player_features(player_id, age_override=age, goals_override=goals, assists_override=assists)
        predicted_value = predict_value(features)
        
        return jsonify({
            'player_id': player_id,
            'predicted_value': predicted_value,
            'features': features
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/top-players')
def api_top_players():
    position = request.args.get('position', 'All')
    limit = request.args.get('limit', 20, type=int)
    players = get_top_players(position, limit)
    return jsonify({'players': players})

@app.route('/api/compare')
def api_compare():
    id1 = request.args.get('player1', type=int)
    id2 = request.args.get('player2', type=int)
    
    if not id1 or not id2:
        return jsonify({'error': 'Deux player_id requis'})
    
    try:
        f1 = prepare_player_features(id1)
        f2 = prepare_player_features(id2)
        v1 = predict_value(f1)
        v2 = predict_value(f2)
        
        return jsonify({
            'player1': {'features': f1, 'predicted_value': v1},
            'player2': {'features': f2, 'predicted_value': v2}
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("🚀 Serveur démarré sur http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
