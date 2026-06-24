# Phase 6 — Observabilité et Monitoring du modèle ML

## Vue d'ensemble

Système complet pour tracker les prédictions en production, détecter les dérives de performance,
et monitorer la santé du modèle.

---

## Architecture

### 1. **observability.py** — Logging des prédictions

Enregistre dans une base SQLite :
- Chaque prédiction (date, montant prédit, montant réel)
- L'erreur associée
- Les features utilisées
- Les métadonnées du modèle

**Utilisation** :
```python
from src.observability import log_prediction

log_prediction(
    date_prediction="2026-06-24",
    use_case="supermarche",
    montant_predit=1_850_000,
    montant_reel=1_920_000,
    alerte_pic=False,
    features={"jour_semaine": 3, "mois": 6},
    model_version="v2"
)
```

Après 24h, on peut comparer avec la réalité et calculer les métriques :
```python
from src.observability import compute_daily_metrics

metrics = compute_daily_metrics(use_case="supermarche", date_str="2026-06-24")
# Retourne : {MAPE, MAE, RMSE, nb_predictions}
```

### 2. **monitoring.py** — Détection des dérives

Classe `ModelMonitor` qui vérifie automatiquement :

| Check | Détecte | Alerte si |
|-------|---------|-----------|
| **Performance Drift** | MAPE se dégrade | +20% vs baseline |
| **Data Drift** | Distribution d'entrées change | p-value < 0.05 (Kolmogorov-Smirnov) |
| **Prediction Coverage** | Prédictions manquantes | < 50% de couverture |
| **Prediction Errors** | Outliers (erreur > 50%) | > 20% d'outliers |

**Utilisation** :
```python
from src.monitoring import ModelMonitor

monitor = ModelMonitor("supermarche", baseline_mape=20.0)
health = monitor.run_full_check()
# Retourne : {"health": "healthy|degraded|critical", "checks": {...}}
```

### 3. **monitoring_dashboard.py** — Dashboard Streamlit

Visualisation en temps réel de :
- Santé globale du modèle (🟢 healthy, 🟡 degraded, 🔴 critical)
- Historique MAPE, MAE, RMSE
- Prédictions récentes et leurs erreurs
- Alertes générées

**Lancer le dashboard** :
```bash
streamlit run app/monitoring_dashboard.py
```

---

## Base de données

Fichier : `data/processed/predictions.db` (SQLite)

### Tables

#### `predictions`
Chaque prédiction effectuée en production.

| Colonne | Type | Description |
|---------|------|-------------|
| id | INTEGER | Clé primaire |
| timestamp | TEXT | Quand la prédiction a été faite |
| date_prediction | TEXT | Pour quel jour (YYYY-MM-DD) |
| use_case | TEXT | supermarche, restaurant, etc |
| montant_predit | REAL | Montant prédit |
| montant_reel | REAL | Montant réel (si connu) |
| erreur_pct | REAL | \|(prédit - réel) / réel\| * 100 |
| alerte_pic | INTEGER | Booléen : pic détecté? |
| features_json | TEXT | Features en JSON |
| model_version | TEXT | v1, v2, etc |

#### `metrics`
Métriques quotidiennes agrégées.

| Colonne | Type | Description |
|---------|------|-------------|
| id | INTEGER | Clé primaire |
| date_calcul | TEXT | Date du calcul |
| use_case | TEXT | Commerce |
| mape | REAL | MAPE % |
| mae | REAL | MAE |
| rmse | REAL | RMSE |
| nb_predictions | INTEGER | Nombre de prédictions ce jour |
| model_version | TEXT | v1, v2, etc |

#### `alerts`
Alertes générées par le monitor.

| Colonne | Type | Description |
|---------|------|-------------|
| id | INTEGER | Clé primaire |
| timestamp | TEXT | Quand l'alerte a été générée |
| use_case | TEXT | Commerce affecté |
| type_alerte | TEXT | model_drift, data_drift, etc |
| message | TEXT | Description lisible |
| metric_name | TEXT | Quelle métrique (ex: mape) |
| metric_value | REAL | Valeur observée |
| seuil | REAL | Seuil d'alerte |

---

## Workflow de production

### Phase 1 : Prédiction quotidienne

Après chaque prédiction faite en production :

```python
# dashboard.py ou autre service
from src.observability import log_prediction

log_prediction(
    date_prediction=date,
    use_case=use_case,
    montant_predit=prediction,
    montant_reel=None,  # On ne sait pas encore
    alerte_pic=(pic_classifier.predict([features])[0] == 1),
    features=features_dict,
)
```

### Phase 2 : Enrichissement avec réalité (24h+)

Une fois les ventes du jour connues, mettre à jour `montant_reel` :

```sql
UPDATE predictions
SET montant_reel = ?
WHERE date_prediction = ?
```

### Phase 3 : Calcul des métriques (quotidien)

```python
from src.observability import compute_daily_metrics

for use_case in ["supermarche", "restaurant", "mobile_money", "grossiste"]:
    compute_daily_metrics(use_case, date_str=today)
```

### Phase 4 : Health check (quotidien ou hebdo)

```python
from src.monitoring import ModelMonitor

for use_case in use_cases:
    monitor = ModelMonitor(use_case, baseline_mape=20)
    health = monitor.run_full_check()
    
    if health["health"] == "critical":
        send_alert_to_admin(health)
        # Trigger re-training?
```

---

## Seuils d'alerte recommandés

| Métrique | OK | Warning | Critical |
|----------|----|---------| ---------|
| **MAPE** | < 15% | 15-25% | > 25% |
| **Data Drift (p-value)** | > 0.05 | 0.01-0.05 | < 0.01 |
| **Coverage** | > 90% | 50-90% | < 50% |
| **Outlier errors** | < 10% | 10-20% | > 20% |

---

## Test du système

Simuler 30 jours de prédictions :

```bash
python3 test_observability.py
```

Cela génère :
- 30 prédictions en base de données
- 7 jours de métriques
- 1 diagnostic de santé
- 1 alerte critique (basé sur les données de test)

---

## Intégration avec l'API/Dashboard

Dans `app/dashboard.py` ou une API:

```python
from src.observability import log_prediction
from src.monitoring import ModelMonitor

# Après une prédiction
log_prediction(
    date_prediction=date,
    use_case=use_case,
    montant_predit=pred_value,
    alerte_pic=pic_alert,
    features=features_dict,
)

# Afficher la santé dans l'UI
monitor = ModelMonitor(use_case)
health = monitor.run_full_check()

if health["health"] == "critical":
    st.error(f"⚠️ Modèle dégradé : {health['checks']['performance_drift']['message']}")
```

---

## Prochaines étapes

1. **Phase 7 (Security)** :
   - Valider les entrées (déterminer si une prédiction est anormale)
   - Bloquer les prédictions impossibles
   - Monitorer les tentatives d'abus

2. **Améliorations** :
   - Dashboards per-feature (quelles features changent le plus?)
   - Auto-retraining si dérive détectée
   - Notifications Slack/Email des alertes critiques
   - Export des métriques pour reporting

---

## Fichiers

| Fichier | Rôle |
|---------|------|
| `src/observability.py` | Logging SQLite, métriques |
| `src/monitoring.py` | Détection des dérives |
| `app/monitoring_dashboard.py` | Dashboard Streamlit |
| `test_observability.py` | Test du système |
| `data/processed/predictions.db` | Base SQLite (créée auto) |
