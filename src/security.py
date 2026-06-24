"""
Sécurité du modèle ML — Validation des entrées, détection d'anomalies, filtering de sortie.
Protège contre les injections, les exploitations et les prédictions impossibles.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple

from src.observability import log_alert


class InputValidator:
    """Valide les entrées avant prédiction."""

    def __init__(self, use_case, historical_data_df=None):
        """
        Args:
            use_case : nom du commerce
            historical_data_df : DataFrame historique pour calculer les stats
        """
        self.use_case = use_case
        self.historical_data = historical_data_df

        # Calculer les stats pour détecter les anomalies
        if historical_data_df is not None and "montant_total" in historical_data_df.columns:
            self.mean_sales = historical_data_df["montant_total"].mean()
            self.std_sales = historical_data_df["montant_total"].std()
            self.min_sales = historical_data_df["montant_total"].min()
            self.max_sales = historical_data_df["montant_total"].max()
        else:
            self.mean_sales = 1_000_000
            self.std_sales = 500_000
            self.min_sales = 100_000
            self.max_sales = 10_000_000

    def validate_features(self, features: Dict) -> Tuple[bool, str]:
        """
        Valide les features avant prédiction.
        Retourne (is_valid, error_message).
        """
        # Vérifier les types
        required_fields = ["jour_semaine", "jour_mois", "mois"]

        for field in required_fields:
            if field not in features:
                return False, f"Champ manquant: {field}"
            if not isinstance(features[field], (int, float, np.integer)):
                return False, f"Type invalide pour {field}: attendu numérique, reçu {type(features[field])}"

        # Vérifier les ranges
        if not (0 <= features.get("jour_semaine", -1) <= 6):
            return False, "jour_semaine doit être entre 0-6"

        if not (1 <= features.get("jour_mois", 0) <= 31):
            return False, "jour_mois doit être entre 1-31"

        if not (1 <= features.get("mois", 0) <= 12):
            return False, "mois doit être entre 1-12"

        # Pas d'injection SQL/code
        for key, value in features.items():
            if isinstance(value, str):
                if any(char in str(value).lower() for char in ["select", "insert", "delete", "exec", "import"]):
                    return False, f"Contenu suspect dans {key}"

        return True, ""

    def validate_prediction_date(self, date_str: str) -> Tuple[bool, str]:
        """Valide que la date de prédiction est raisonnable."""
        try:
            pred_date = pd.to_datetime(date_str)
        except:
            return False, f"Format de date invalide: {date_str}"

        today = pd.Timestamp.now()

        # Pas de prédiction dans le passé
        if pred_date < today:
            return False, f"Cannot predict for past date: {date_str}"

        # Pas au-delà de 30 jours
        if pred_date > today + timedelta(days=30):
            return False, "Prédiction limitée à 30 jours"

        return True, ""

    def detect_anomaly(self, features: Dict, severity_threshold=3.0) -> Tuple[bool, str]:
        """
        Détecte si les features ressemblent à une tentative d'exploitation.
        Utilise l'écart-type pour identifier les valeurs anormales.
        """
        anomalies = []

        # Vérifier des valeurs trop extrêmes
        if "temperature_max" in features:
            temp = features["temperature_max"]
            if temp < -50 or temp > 60:
                anomalies.append(f"Température extrême: {temp}°C")

        if "precipitation_mm" in features:
            prec = features["precipitation_mm"]
            if prec < 0 or prec > 500:
                anomalies.append(f"Précipitation extrême: {prec}mm")

        if "ventes_j_1" in features:
            ventes = features["ventes_j_1"]
            if abs(ventes - self.mean_sales) > severity_threshold * self.std_sales:
                lower = self.mean_sales - severity_threshold * self.std_sales
                upper = self.mean_sales + severity_threshold * self.std_sales
                anomalies.append(
                    f"Ventes J-1 anormale: {ventes:.0f} (attendre [{lower:.0f}, {upper:.0f}])"
                )

        if anomalies:
            return True, "; ".join(anomalies)

        return False, ""


class OutputFilter:
    """Filtre et valide les prédictions avant exposition."""

    def __init__(self, use_case, historical_mean=None):
        self.use_case = use_case
        self.historical_mean = historical_mean or 1_000_000
        self.min_reasonable = self.historical_mean * 0.01  # Minimum 1% de moyenne
        self.max_reasonable = self.historical_mean * 10   # Maximum 10x moyenne

    def filter_prediction(self, prediction: float) -> Tuple[float, bool, str]:
        """
        Valide et filtre une prédiction.
        Retourne (prediction_filtered, is_valid, warning).
        """
        warning = ""

        # Vérifier les valeurs impossibles
        if prediction < 0:
            return 0, False, "Prédiction négative (impossible)"

        if np.isnan(prediction) or np.isinf(prediction):
            return 0, False, "Prédiction invalide (NaN ou Inf)"

        # Clamp aux limites raisonnables
        if prediction < self.min_reasonable:
            warning = f"Prédiction très basse ({prediction:.0f} < {self.min_reasonable:.0f}), clampée"
            prediction = self.min_reasonable

        if prediction > self.max_reasonable:
            warning = f"Prédiction très haute ({prediction:.0f} > {self.max_reasonable:.0f}), clampée"
            prediction = self.max_reasonable

        return float(prediction), True, warning

    def filter_alert_confidence(self, pic_probability: float) -> Tuple[float, bool]:
        """
        Valide la confiance d'une alerte pic.
        Retourne (confidence_filtered, is_valid).
        """
        if not (0 <= pic_probability <= 1):
            return 0.0, False

        # Ne créer une alerte que si confiance > 60%
        if pic_probability < 0.6:
            return 0.0, False

        return float(pic_probability), True
