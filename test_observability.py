"""
Test d'observabilité — Simule des prédictions et teste le système de monitoring.
"""

import numpy as np
from datetime import datetime, timedelta
from src.observability import log_prediction, init_db, compute_daily_metrics
from src.monitoring import ModelMonitor

np.random.seed(42)

print("=" * 60)
print("  TEST OBSERVABILITY — Simulation de prédictions")
print("=" * 60)

init_db()

# Simuler 30 jours de prédictions pour supermarche
use_case = "supermarche"
base_date = datetime.now() - timedelta(days=30)

print(f"\n[1/3] Enregistrement de 30 jours de prédictions...")

for day in range(30):
    date = base_date + timedelta(days=day)
    date_str = date.strftime("%Y-%m-%d")

    # Simuler des prédictions avec du bruit réaliste
    ventes_reelles = np.random.normal(1_800_000, 400_000)
    ventes_predites = ventes_reelles + np.random.normal(0, ventes_reelles * 0.15)

    ventes_predites = max(0, ventes_predites)

    # 10% de chance d'avoir un pic
    is_peak = np.random.random() < 0.1

    log_prediction(
        date_prediction=date_str,
        use_case=use_case,
        montant_predit=ventes_predites,
        montant_reel=ventes_reelles,
        alerte_pic=is_peak,
        features={"jour_semaine": date.weekday(), "mois": date.month},
        model_version="v2",
    )

print(f"✅ 30 prédictions enregistrées")

print(f"\n[2/3] Calcul des métriques quotidiennes...")

# Calculer les métriques pour les derniers jours
for day in range(-7, 0):
    date = datetime.now() + timedelta(days=day)
    date_str = date.strftime("%Y-%m-%d")

    try:
        metrics = compute_daily_metrics(use_case, date_str=date_str)
        if metrics:
            print(f"  {date_str}: MAPE={metrics['mape']:.1f}% MAE={metrics['mae']:.0f}")
    except Exception as e:
        print(f"  {date_str}: {e}")

print(f"\n[3/3] Diagnostic de santé du modèle...")

monitor = ModelMonitor(use_case, baseline_mape=20.0)
health = monitor.run_full_check()

print(f"\n{'='*60}")
print(f"  RÉSULTATS")
print(f"{'='*60}")
print(f"  Santé globale     : {health['health'].upper()}")
print(f"  Timestamp         : {health['timestamp']}")

for check_name, result in health["checks"].items():
    status = result.get("status", "unknown").upper()
    message = result.get("message", "")
    print(f"\n  {check_name.replace('_', ' ').title()}")
    print(f"    Status  : {status}")
    print(f"    Message : {message}")

print(f"\n{'='*60}")
print(f"✅ Test complété — Données enregistrées dans data/processed/predictions.db")
print(f"   Lancer le dashboard : streamlit run app/monitoring_dashboard.py")
print(f"{'='*60}")
