# ⚽ Football Player Market Value Prediction

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black.svg)](https://flask.palletsprojects.com/)


> 🤖 Application ML complète pour prédire et analyser la valeur marchande des joueurs de football professionnel. Interface web moderne avec comparaison, simulation et projection de carrière.

## ✨ Fonctionnalités

### 🔍 Recherche Intelligente
- Autocomplétion temps réel sur 92,000+ joueurs
- Tri par pertinence (valeur marchande, matchs, sélections)
- Affichage club, position, âge

### 💰 Prédiction de Valeur
- Estimation basée sur 28 features
- Explications IA détaillées (facteurs positifs/négatifs)
- Historique des valeurs avec graphique temporel

### 🔮 Projection de Carrière
- Prédiction de la valeur future jusqu'à 34 ans
- Âge de pic personnalisé selon la position :
  - ⚡ Attaquants : 26 ans
  - 🎯 Milieux : 27 ans
  - 🛡️ Défenseurs : 28 ans
  - 🧤 Gardiens : 29 ans
- Graphique de progression avec point de pic

### ⚖️ Comparateur de Joueurs
- Comparaison côte à côte de 2 joueurs
- Radar chart interactif (6 dimensions)
- Différence de valeur calculée

### 🎮 Simulateur What-If
- Modifier âge, buts, passes décisives
- Voir l'impact en temps réel sur la valeur
- Tester des scénarios de progression

### 🏆 Top Joueurs
- Classement par valeur marchande
- Filtrage par position
- Accès rapide aux fiches joueurs

### 🎨 Interface Moderne
- Design dark/light mode
- Graphiques Chart.js animés
- Export CSV/JSON
- 100% responsive

## 📊 Performance du Modèle

| Métrique | Valeur |
|----------|--------|
| **R² Score** | 0.6626 |
| **MAE** | €950,701 |
| **Algorithme** | GradientBoostingRegressor |
| **Features** | 28 |
| **Joueurs** | 92,671 |

## 🚀 Quickstart

### Prérequis
- Docker & Docker Compose
- Données Kaggle (voir section Données)
- Modèle entraîné (voir section Entraînement)

### Lancement rapide

```bash
# 1. Cloner le repo
git clone https://github.com/CheikhAiLabs/football_player_market.git
cd football_player_market

# 2. Télécharger les données (voir section Données)

# 3. Entraîner le modèle
python src/train_model_v2.py

# 4. Lancer l'application
cd app && docker-compose up --build -d

# 5. Ouvrir http://localhost:5000
```

## 📁 Structure du Projet

```
football_player_market/
├── app/                        # 🌐 Application Web
│   ├── app.py                 # Flask + Interface complète
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── src/                        # 🧠 Code ML
│   ├── data_pipeline.py       # Préparation des données
│   ├── train_model_v2.py      # Entraînement GradientBoosting
│   └── evaluate.py            # Évaluation et métriques
├── data/                       # 📊 Données (non versionnées)
│   ├── player_profiles/
│   ├── player_performances/
│   ├── player_market_value/
│   ├── player_injuries/
│   └── transfer_history/
├── models/                     # 🎯 Modèles entraînés
│   └── football_model.pkl
├── analysis/                   # 📈 Rapports
│   └── evaluation_report.txt
├── tests/
│   └── validation.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 📦 Données

**Source :** [Kaggle Football Datasets](https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets)

| Dataset | Records |
|---------|---------|
| Player Profiles | 92,671 |
| Market Values | 901,429 |
| Performances | 1,878,719 |
| Injuries | 143,195 |
| Transfers | 1,101,440 |

### Téléchargement

```bash
# Installer l'API Kaggle
pip install kaggle

# Configurer les credentials
export KAGGLE_USERNAME=<your_username>
export KAGGLE_KEY=<your_api_key>

# Télécharger
kaggle datasets download -d xfkzujqjvx97n/football-datasets -p data/ --unzip
```

## 🔧 Installation Locale (Développement)

```bash
# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Entraîner le modèle
python src/train_model_v2.py

# Lancer en mode dev
cd app && python app.py
```

## 🐳 Docker (Production)

```bash
cd app

# Build et démarrer
docker-compose up --build -d

# Voir les logs
docker-compose logs -f football-predictor

# Arrêter
docker-compose down

# Rebuild complet
docker-compose down && docker-compose build --no-cache && docker-compose up -d
```

## 🌐 API Endpoints

| Endpoint | Description | Exemple |
|----------|-------------|---------|
| `GET /` | Interface web complète | `http://localhost:5000` |
| `GET /api/search` | Recherche joueurs | `/api/search?name=mbappe` |
| `GET /api/predict` | Prédiction + projection | `/api/predict?player_id=342229` |
| `GET /api/simulate` | Simulation what-if | `/api/simulate?player_id=342229&age=30&goals=400` |
| `GET /api/top-players` | Classement | `/api/top-players?position=Forward` |

### Exemples cURL

```bash
# Rechercher un joueur
curl "http://localhost:5000/api/search?name=haaland"

# Prédire la valeur avec projection future
curl "http://localhost:5000/api/predict?player_id=418560"

# Simuler à 28 ans avec 300 buts
curl "http://localhost:5000/api/simulate?player_id=418560&age=28&goals=300&assists=80"

# Top attaquants
curl "http://localhost:5000/api/top-players?position=Forward"
```

## 🧠 Features du Modèle

Le modèle utilise **28 features** réparties en catégories :

| Catégorie | Features |
|-----------|----------|
| **Profil** | `age`, `height`, `years_at_club`, `contract_length` |
| **Performance** | `total_goals`, `total_assists`, `total_minutes`, `total_appearances` |
| **Ratios** | `goals_per_match`, `assists_per_match`, `goals_per_90`, `assists_per_90` |
| **Âge** | `age_squared`, `age_from_peak` (pic à 27 ans) |
| **Blessures** | `total_injury_days`, `total_games_missed`, `injury_count`, `injury_rate` |
| **Transferts** | `total_transfer_fees`, `avg_transfer_fee`, `transfer_count` |
| **Sélections** | `national_matches`, `national_goals`, `national_goal_ratio` |
| **Encodés** | `position_encoded`, `main_position_encoded`, `foot_encoded` |

## 📋 Requirements

**Système :**
- Python 3.11+
- Docker & Docker Compose
- ~8GB RAM recommandé
- ~500MB espace disque (données)

**Python :**
```txt
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
joblib>=1.3.0
flask>=3.0.0
```

## 🎯 Exemples de Prédictions

| Joueur | Âge | Position | Valeur Prédite | Pic Estimé |
|--------|-----|----------|----------------|------------|
| Kylian Mbappé | 27 | Attaquant | €90M | - (passé) |
| Erling Haaland | 25 | Attaquant | €150M | €180M (26 ans) |
| Jude Bellingham | 22 | Milieu | €120M | €160M (27 ans) |
| Ibrahim Mbaye | 17 | Attaquant | €3.3M | €25M+ (26 ans) |

## 🔮 Roadmap

- [x] Prédiction de valeur avec ML
- [x] Interface web moderne
- [x] Comparateur de joueurs
- [x] Simulateur what-if
- [x] Projection de carrière
- [x] Graphiques historiques
- [x] Export CSV/JSON
- [ ] API authentifiée
- [ ] Intégration XGBoost/LightGBM
- [ ] Données en temps réel
- [ ] App mobile

## 🤝 Contribution

Les contributions sont les bienvenues ! 

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## � Auteur

**Cheikh-GPT**

---

<p align="center">
  <b>Développé avec ❤️ et beaucoup de données football</b><br>
  ⭐ Star ce repo si tu le trouves utile !
</p>
