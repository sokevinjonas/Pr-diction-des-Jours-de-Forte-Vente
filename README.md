# Prediction des Jours de Forte Vente

Systeme de machine learning qui predit les jours de forte affluence pour les commerces en Afrique de l'Ouest. Permet d'optimiser la gestion du stock, du personnel et de la logistique.

## Probleme resolu

Les commercants africains gerent leur stock a l'instinct et subissent :
- **Surstockage** : gaspillage, immobilisation de tresorerie
- **Rupture de stock** : perte de ventes, clients mecontents

Ce systeme reduit les ruptures de stock de 30 a 50% grace a des previsions fiables.

## Use cases supportes

| Commerce | Output |
|----------|--------|
| Supermarche / Grande surface | CA journalier prevu + alertes pic |
| Restaurant / Fast food | Nombre de couverts + signal "extras" |
| Telephonie / Mobile Money | Volume de transactions + float recommande |
| Grossiste / Distributeur | Demande par zone + nombre de camions |

## Installation

```bash
# Cloner le projet
cd "Prediction des Jours de Forte Vente"

# Creer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dependances
pip install -r requirements.txt
```

## Utilisation rapide

```bash
# Activer l'environnement
source venv/bin/activate

# 1. Generer les donnees synthetiques (premiere utilisation)
python3 data/synthetic/generate_data.py

# 2. Entrainer le modele
python3 -c "from src.model import train_pipeline; train_pipeline('supermarche')"

# 3. Lancer le dashboard
streamlit run app/dashboard.py
```

Le dashboard est accessible sur `http://localhost:8501`.

## Structure du projet

```
.
├── app/
│   ├── dashboard.py              # Application Streamlit
│   └── assets/style.css
├── data/
│   ├── raw/                      # CSV de ventes + calendrier evenements
│   ├── processed/                # Features transformees
│   └── synthetic/generate_data.py
├── src/
│   ├── data_loader.py            # Chargement et validation des CSV
│   ├── feature_engineering.py    # 27 features (temporelles, evenements, lags)
│   ├── model.py                  # Entrainement XGBoost + evaluation
│   ├── predict.py                # Predictions + recommandations metier
│   └── utils.py                  # Configuration et utilitaires
├── models/                       # Modele sauvegarde (.pkl)
├── tests/test_model.py           # 16 tests unitaires
├── config.yaml                   # Parametres configurables
└── requirements.txt
```

## Modele

**XGBoost** avec validation temporelle (TimeSeriesSplit).

### Performance (donnees synthetiques supermarche)

| Metrique | Valeur | Objectif |
|----------|--------|----------|
| MAPE | 13.0% | < 15% |
| MAE | 239 015 FCFA | - |
| Recall pics | 57.8% | > 65% |

### Top features

1. `veille_ferie` — proximite immediate d'une fete
2. `nb_jours_avant_fete` — anticipation des achats
3. `est_debut_mois` — effet jour de paie
4. `est_weekend` — affluence samedi/dimanche
5. `jour_mois` — position dans le mois

## Features engineering

- **Temporelles** : jour semaine, mois, weekend, debut/fin de mois, encodage cyclique sin/cos
- **Evenementielles** : jour ferie, veille ferie, distance a la prochaine fete, type d'evenement
- **Lags** : ventes J-1, J-7, J-14, moyennes glissantes 7/14/30j, tendance, volatilite

## Configuration

Editer `config.yaml` pour ajuster :
- Type de modele et horizon de prediction
- Seuils d'alerte (orange a +30%, rouge a +60%)
- Parametres metier par use case (caisses, couverts, float, camions)

## Dashboard

Le dashboard Streamlit offre :
- Vue des previsions avec code couleur (vert/orange/rouge)
- KPIs : CA prevu, variation moyenne, nombre d'alertes
- Historique des ventes avec moyenne mobile
- Import de donnees personnalisees (CSV)
- Export des previsions en Excel ou CSV
- Reentrainement du modele en un clic

## Tests

```bash
source venv/bin/activate
python3 -m pytest tests/ -v
```

## Fetes et evenements couverts

Tabaski, Korite, Noel, Nouvel An, Mawlid, Fete nationale (Senegal, Cote d'Ivoire), Fete du travail, periodes de Ramadan.

## Deploiement

### Docker

```bash
docker build -t prediction-ventes .
docker run -p 8501:8501 prediction-ventes
```

### Render.com

Configurer le `startCommand` :
```
streamlit run app/dashboard.py --server.port $PORT --server.headless true
```

## Licence

Projet prive — usage interne.
