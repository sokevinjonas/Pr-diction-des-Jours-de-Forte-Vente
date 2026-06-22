"""
Module de prédiction : génère les prévisions et recommandations.
Produit un JSON structuré avec alertes et actions concrètes.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

from src.utils import PROJECT_ROOT, load_config, niveau_alerte
from src.data_loader import load_ventes
from src.feature_engineering import build_features, add_features_temporelles, add_features_evenements


MODELS_DIR = PROJECT_ROOT / "models"


def predict_next_days(horizon=7, use_case="supermarche", filepath=None):
    """
    Prédit les ventes pour les N prochains jours.

    Retourne un DataFrame avec : date, prediction, variation, niveau_alerte, message.
    """
    config = load_config()
    seuils = config["seuils_alerte"]

    # Charger le modèle
    model_data = joblib.load(MODELS_DIR / "xgboost_model.pkl")
    model = model_data["model"]
    feature_cols = model_data["features"]

    # Charger les données historiques pour calculer les lags
    df_hist = load_ventes(use_case=use_case, filepath=filepath)

    # Calculer la moyenne récente (référence)
    moyenne_recente = df_hist["montant_total"].tail(30).mean()

    # Générer les dates futures
    derniere_date = df_hist["date"].max()
    dates_futures = pd.date_range(
        start=derniere_date + timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    predictions = []

    for i, date_pred in enumerate(dates_futures):
        # Construire un mini-DataFrame avec la date à prédire
        row = pd.DataFrame({"date": [date_pred], "montant_total": [np.nan]})

        # Concaténer avec l'historique pour calculer les features
        df_temp = pd.concat([df_hist, row], ignore_index=True)
        df_temp = build_features(df_temp, target_col="montant_total")

        # Prendre la dernière ligne (celle qu'on prédit)
        if df_temp.empty:
            continue

        last_row = df_temp.iloc[[-1]]
        features_dispo = [c for c in feature_cols if c in last_row.columns]

        X_pred = last_row[features_dispo]

        # Prédiction
        pred_montant = float(model.predict(X_pred)[0])
        pred_montant = max(0, pred_montant)

        # Variation par rapport à la normale
        variation = (pred_montant - moyenne_recente) / moyenne_recente

        # Niveau d'alerte
        alerte = niveau_alerte(variation, seuils)

        # Recommandations contextuelles
        message = _generer_message(variation, alerte, date_pred, use_case, config)

        predictions.append({
            "date": date_pred.strftime("%Y-%m-%d"),
            "jour": _jour_francais(date_pred.weekday()),
            "prediction": round(pred_montant),
            "variation": f"{variation:+.0%}",
            "variation_pct": round(variation * 100, 1),
            "niveau_alerte": alerte,
            "message": message,
            "recommandations": _generer_recommandations(
                variation, use_case, config, pred_montant
            ),
        })

        # Ajouter la prédiction à l'historique pour les lags suivants
        df_hist = pd.concat([
            df_hist,
            pd.DataFrame({"date": [date_pred], "montant_total": [pred_montant]})
        ], ignore_index=True)

    return pd.DataFrame(predictions)


def predict_single_day(date_str, use_case="supermarche"):
    """Prédit pour un seul jour donné. Retourne un dict JSON-ready."""
    date_cible = pd.Timestamp(date_str)
    df_hist = load_ventes(use_case=use_case)
    derniere_date = df_hist["date"].max()

    horizon = (date_cible - derniere_date).days
    if horizon <= 0:
        raise ValueError(
            f"La date {date_str} est dans le passé ou le présent. "
            f"Dernière date disponible : {derniere_date.date()}"
        )

    df_pred = predict_next_days(horizon=horizon, use_case=use_case)
    row = df_pred[df_pred["date"] == date_str]

    if row.empty:
        raise ValueError(f"Impossible de prédire pour {date_str}")

    return row.iloc[0].to_dict()


def _jour_francais(weekday):
    jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    return jours[weekday]


def _generer_message(variation, alerte, date, use_case, config):
    """Génère un message explicatif adapté au contexte."""
    if alerte == "ROUGE":
        return (
            f"Pic de ventes attendu ({variation:+.0%}). "
            f"Renforcer le stock et le personnel."
        )
    elif alerte == "ORANGE":
        return (
            f"Hausse moderee prevue ({variation:+.0%}). "
            f"Verifier les stocks des produits phares."
        )
    else:
        if variation < -0.15:
            return f"Journee calme attendue ({variation:+.0%}). Possibilite de reduction d'effectif."
        return "Journee normale prevue."


def _generer_recommandations(variation, use_case, config, montant_prevu):
    """Génère des recommandations métier concrètes."""
    uc_config = config["use_cases"].get(use_case, {})
    reco = {"stock_supplementaire": False, "personnel_extra": 0}

    if variation >= 0.30:
        reco["stock_supplementaire"] = True

        if use_case == "supermarche":
            nb_caisses = uc_config.get("nb_caisses_normal", 5)
            reco["caisses_recommandees"] = min(nb_caisses * 2, int(nb_caisses * (1 + variation)))
            reco["personnel_extra"] = max(1, int(variation * 5))

        elif use_case == "restaurant":
            couverts_normal = uc_config.get("couverts_normal", 80)
            reco["couverts_prevus"] = int(couverts_normal * (1 + variation))
            reco["personnel_extra"] = max(1, int(variation * 3))

        elif use_case == "mobile_money":
            float_pic = uc_config.get("float_pic", 750000)
            reco["float_recommande"] = int(float_pic * (1 + variation * 0.5))

        elif use_case == "grossiste":
            nb_camions = uc_config.get("nb_camions_normal", 3)
            reco["camions_recommandes"] = min(nb_camions * 2, int(nb_camions * (1 + variation)))

    return reco
