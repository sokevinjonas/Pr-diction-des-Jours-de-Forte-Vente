"""
Feature engineering pour la prédiction des ventes.
Crée les variables temporelles, événementielles et de lag.
"""

import pandas as pd
import numpy as np
from src.data_loader import load_evenements


def add_features_temporelles(df):
    """Ajoute les features dérivées de la date."""
    df = df.copy()
    df["jour_semaine"] = df["date"].dt.weekday
    df["jour_mois"] = df["date"].dt.day
    df["semaine_mois"] = (df["date"].dt.day - 1) // 7 + 1
    df["mois"] = df["date"].dt.month
    df["trimestre"] = df["date"].dt.quarter
    df["est_weekend"] = df["jour_semaine"].isin([5, 6]).astype(int)
    df["est_debut_mois"] = (df["jour_mois"] <= 5).astype(int)
    df["est_fin_mois"] = (df["jour_mois"] >= 26).astype(int)
    df["jour_annee"] = df["date"].dt.dayofyear

    # Encodage cyclique du jour de la semaine et du mois
    df["jour_semaine_sin"] = np.sin(2 * np.pi * df["jour_semaine"] / 7)
    df["jour_semaine_cos"] = np.cos(2 * np.pi * df["jour_semaine"] / 7)
    df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12)
    df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12)

    return df


def add_features_evenements(df):
    """Ajoute les features liées aux fêtes et événements."""
    df = df.copy()
    df_evt = load_evenements()

    if df_evt.empty:
        df["est_jour_ferie"] = 0
        df["veille_ferie"] = 0
        df["nb_jours_avant_fete"] = 999
        df["nb_jours_apres_fete"] = 999
        df["type_evenement"] = "aucun"
        return df

    dates_fetes = set(df_evt["date"].dt.date)

    df["est_jour_ferie"] = df["date"].dt.date.isin(dates_fetes).astype(int)

    # Veille de fête
    veilles = {d - pd.Timedelta(days=1) for d in df_evt["date"]}
    df["veille_ferie"] = df["date"].dt.date.isin(
        {v.date() if hasattr(v, 'date') else v for v in veilles}
    ).astype(int)

    # Distance à la prochaine fête et à la dernière fête
    dates_fetes_sorted = sorted(df_evt["date"].unique())

    nb_avant = []
    nb_apres = []
    types_evt = []

    for d in df["date"]:
        # Jours avant la prochaine fête
        futures = [f for f in dates_fetes_sorted if f >= d]
        if futures:
            nb_avant.append((futures[0] - d).days)
        else:
            nb_avant.append(999)

        # Jours après la dernière fête
        passees = [f for f in dates_fetes_sorted if f <= d]
        if passees:
            nb_apres.append((d - passees[-1]).days)
        else:
            nb_apres.append(999)

        # Type d'événement du jour
        evt_jour = df_evt[df_evt["date"] == d]
        if not evt_jour.empty:
            types_evt.append(evt_jour.iloc[0]["type"])
        else:
            types_evt.append("aucun")

    df["nb_jours_avant_fete"] = nb_avant
    df["nb_jours_apres_fete"] = nb_apres
    df["type_evenement"] = types_evt

    return df


def add_features_lag(df, target_col="montant_total"):
    """
    Ajoute les features de lag (valeurs passées).
    IMPORTANT : ne pas utiliser ces features pour les jours futurs en production.
    """
    df = df.copy()

    df["ventes_j_1"] = df[target_col].shift(1)
    df["ventes_j_2"] = df[target_col].shift(2)
    df["ventes_j_7"] = df[target_col].shift(7)
    df["ventes_j_14"] = df[target_col].shift(14)

    # Moyennes glissantes
    df["ventes_moy_7j"] = df[target_col].shift(1).rolling(window=7, min_periods=1).mean()
    df["ventes_moy_14j"] = df[target_col].shift(1).rolling(window=14, min_periods=1).mean()
    df["ventes_moy_30j"] = df[target_col].shift(1).rolling(window=30, min_periods=1).mean()

    # Écart-type glissant (volatilité récente)
    df["ventes_std_7j"] = df[target_col].shift(1).rolling(window=7, min_periods=1).std()

    # Tendance sur 7 jours (pente)
    df["tendance_7j"] = df[target_col].shift(1).rolling(window=7, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0,
        raw=True
    )

    # Ratio par rapport à la moyenne (détecte les anomalies)
    df["ratio_vs_moy_30j"] = df[target_col].shift(1) / df["ventes_moy_30j"]

    return df


def build_features(df, target_col="montant_total"):
    """Pipeline complet de feature engineering."""
    df = add_features_temporelles(df)
    df = add_features_evenements(df)
    df = add_features_lag(df, target_col=target_col)

    # Supprimer les premières lignes sans lag suffisant
    df = df.dropna(subset=["ventes_j_14"]).reset_index(drop=True)

    return df


def get_feature_columns():
    """Retourne la liste des colonnes de features pour le modèle."""
    return [
        "jour_semaine", "jour_mois", "semaine_mois", "mois", "trimestre",
        "est_weekend", "est_debut_mois", "est_fin_mois", "jour_annee",
        "jour_semaine_sin", "jour_semaine_cos", "mois_sin", "mois_cos",
        "est_jour_ferie", "veille_ferie",
        "nb_jours_avant_fete", "nb_jours_apres_fete",
        "ventes_j_1", "ventes_j_2", "ventes_j_7", "ventes_j_14",
        "ventes_moy_7j", "ventes_moy_14j", "ventes_moy_30j",
        "ventes_std_7j", "tendance_7j", "ratio_vs_moy_30j",
    ]
