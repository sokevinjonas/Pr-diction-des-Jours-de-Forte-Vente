"""
Monitoring du modèle ML — Détecte les dérives de performance (model drift).
Alerte si le modèle se dégrade ou si les données changent.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.observability import (
    get_metrics_history, get_predictions, log_alert
)


class ModelMonitor:
    """Monitore la qualité et la stabilité du modèle en production."""

    def __init__(self, use_case, baseline_mape=None, baseline_mae=None):
        """
        Args:
            use_case : nom du commerce
            baseline_mape : MAPE de référence (seuil d'alerte)
            baseline_mae : MAE de référence
        """
        self.use_case = use_case
        self.baseline_mape = baseline_mape or 20.0  # 20% par défaut
        self.baseline_mae = baseline_mae

    def check_performance_drift(self, days=7):
        """
        Vérifie si les performances se dégradent récemment.
        Compare les derniers jours avec la moyenne historique.
        """
        metrics = get_metrics_history(self.use_case, days=90)

        if len(metrics) < 7:
            return {"status": "insufficient_data", "message": "Pas assez de données"}

        # Séparer ancien (premier 60j) vs récent (derniers 7j)
        old_metrics = metrics.iloc[:-7] if len(metrics) > 7 else metrics.head(1)
        recent_metrics = metrics.iloc[-7:] if len(metrics) >= 7 else metrics

        old_mape = old_metrics["mape"].mean()
        recent_mape = recent_metrics["mape"].mean()

        # Dérive = augmentation de > 5% points
        drift_pct = ((recent_mape - old_mape) / old_mape * 100) if old_mape > 0 else 0

        status = "ok"
        if recent_mape > self.baseline_mape:
            status = "warning"
        if drift_pct > 20:  # +20% de dégradation = alerte
            status = "critical"

        return {
            "status": status,
            "old_mape": round(old_mape, 2),
            "recent_mape": round(recent_mape, 2),
            "drift_pct": round(drift_pct, 1),
            "message": self._status_message(status, recent_mape, drift_pct),
        }

    def check_data_drift(self, recent_predictions_df, days=7):
        """
        Détecte si les données d'entrée changent significativement.
        Compare la distribution récente vs ancienne.
        """
        if recent_predictions_df.empty:
            return {"status": "ok", "message": "Pas de données récentes"}

        # Récupérer les montants réels historiques
        all_preds = get_predictions(self.use_case, days=90, include_nulls=True)

        if len(all_preds) < 14:
            return {"status": "insufficient_data", "message": "Historique insuffisant"}

        # Diviser en ancien vs récent
        old_preds = all_preds[all_preds["timestamp"] < (
            datetime.now() - timedelta(days=days)
        )]["montant_reel"].dropna()
        recent_preds = all_preds[all_preds["timestamp"] >= (
            datetime.now() - timedelta(days=days)
        )]["montant_reel"].dropna()

        if len(old_preds) < 5 or len(recent_preds) < 5:
            return {"status": "ok", "message": "Données insuffisantes"}

        # Tests statistiques : Kolmogorov-Smirnov
        from scipy.stats import ks_2samp

        stat, p_value = ks_2samp(old_preds, recent_preds)

        # Si p < 0.05 → distributions significativement différentes
        status = "ok"
        if p_value < 0.05:
            status = "warning"
        if p_value < 0.01:
            status = "critical"

        return {
            "status": status,
            "p_value": round(p_value, 4),
            "old_mean": round(old_preds.mean()),
            "recent_mean": round(recent_preds.mean()),
            "message": self._data_drift_message(status, p_value, old_preds.mean(), recent_preds.mean()),
        }

    def check_prediction_coverage(self, days=7):
        """Vérifie qu'on n'a pas trop de prédictions manquantes."""
        preds = get_predictions(self.use_case, days=days, include_nulls=False)

        if len(preds) == 0:
            return {
                "status": "critical",
                "message": "Aucune prédiction enregistrée!",
                "coverage_pct": 0,
            }

        expected = days
        coverage_pct = len(preds) / expected * 100

        status = "ok"
        if coverage_pct < 80:
            status = "warning"
        if coverage_pct < 50:
            status = "critical"

        return {
            "status": status,
            "coverage_pct": round(coverage_pct, 1),
            "nb_predictions": len(preds),
            "expected": expected,
            "message": f"Couverture: {coverage_pct:.0f}% ({len(preds)}/{expected} jours)",
        }

    def check_prediction_errors(self, days=7, max_error_pct=50):
        """Détecte les prédictions avec erreurs anormalement élevées."""
        preds = get_predictions(self.use_case, days=days, include_nulls=False)

        if preds.empty:
            return {"status": "ok", "message": "Pas de données", "outliers": 0}

        # Identifier les outliers (erreur > max_error_pct)
        outliers = preds[preds["erreur_pct"] > max_error_pct]

        status = "ok"
        if len(outliers) > 0:
            status = "warning"
        if len(outliers) / len(preds) > 0.2:  # > 20% d'erreurs élevées
            status = "critical"

        return {
            "status": status,
            "outliers": len(outliers),
            "total": len(preds),
            "outlier_pct": round(len(outliers) / len(preds) * 100, 1),
            "message": f"{len(outliers)} prédictions avec erreur > {max_error_pct}%",
        }

    def run_full_check(self, recent_predictions_df=None):
        """Lance tous les checks et compile les résultats."""
        checks = {
            "performance_drift": self.check_performance_drift(),
            "prediction_coverage": self.check_prediction_coverage(),
            "prediction_errors": self.check_prediction_errors(),
        }

        if recent_predictions_df is not None and not recent_predictions_df.empty:
            checks["data_drift"] = self.check_data_drift(recent_predictions_df)

        # Alerte globale si un check est critique
        has_critical = any(c.get("status") == "critical" for c in checks.values())
        has_warning = any(c.get("status") == "warning" for c in checks.values())

        health = "healthy"
        if has_critical:
            health = "critical"
        elif has_warning:
            health = "degraded"

        # Logger les alertes critiques
        for check_name, check_result in checks.items():
            if check_result.get("status") == "critical":
                log_alert(
                    self.use_case,
                    "model_drift",
                    f"{check_name}: {check_result.get('message')}",
                    metric_name=check_name,
                )

        return {
            "health": health,
            "timestamp": datetime.now().isoformat(),
            "use_case": self.use_case,
            "checks": checks,
        }

    def _status_message(self, status, mape, drift_pct):
        if status == "critical":
            return f"ALERTE: MAPE {mape:.1f}% (+{drift_pct:.0f}% vs baseline)"
        elif status == "warning":
            return f"Attention: MAPE {mape:.1f}% (seuil: {self.baseline_mape}%)"
        else:
            return f"OK: MAPE {mape:.1f}%"

    def _data_drift_message(self, status, p_value, old_mean, recent_mean):
        change_pct = abs(recent_mean - old_mean) / old_mean * 100 if old_mean > 0 else 0

        if status == "critical":
            return f"ALERTE: Distribution de données changée (p={p_value:.4f}, {change_pct:.0f}% variation)"
        elif status == "warning":
            return f"Attention: Distribution sensiblement différente (p={p_value:.4f})"
        else:
            return "Distributions stables"
