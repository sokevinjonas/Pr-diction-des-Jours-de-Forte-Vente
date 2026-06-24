"""
Features météo via l'API open-meteo.com (gratuite, pas de clé nécessaire).
Ajoute température et précipitations aux données de ventes.
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path

from src.utils import PROJECT_ROOT

METEO_CACHE = PROJECT_ROOT / "data" / "processed" / "meteo_cache.csv"


# Coordonnées des villes principales
VILLES = {
    "SN": {"name": "Dakar", "lat": 14.6937, "lon": -17.4441},
    "CI": {"name": "Abidjan", "lat": 5.3600, "lon": -4.0083},
    "BF": {"name": "Ouagadougou", "lat": 12.3714, "lon": -1.5197},
    "EC": {"name": "Quito", "lat": -0.1807, "lon": -78.4678},
    "DEFAULT": {"name": "Dakar", "lat": 14.6937, "lon": -17.4441},
}


def fetch_meteo(date_start, date_end, pays="SN"):
    """
    Récupère les données météo historiques depuis open-meteo.com.
    Retourne un DataFrame avec : date, temperature_max, temperature_min, precipitation_mm.
    """
    ville = VILLES.get(pays, VILLES["DEFAULT"])

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": ville["lat"],
        "longitude": ville["lon"],
        "start_date": date_start,
        "end_date": date_end,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  [METEO] Erreur API : {e}")
        return None

    if "daily" not in data:
        return None

    df_meteo = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "temperature_max": data["daily"]["temperature_2m_max"],
        "temperature_min": data["daily"]["temperature_2m_min"],
        "precipitation_mm": data["daily"]["precipitation_sum"],
    })

    return df_meteo


def get_meteo_features(df, pays="SN"):
    """
    Ajoute les features météo à un DataFrame de ventes.
    Utilise un cache local pour éviter de rappeler l'API.
    """
    date_min = df["date"].min().strftime("%Y-%m-%d")
    date_max = df["date"].max().strftime("%Y-%m-%d")

    # Tenter le cache
    df_meteo = _load_cache(date_min, date_max)

    if df_meteo is None:
        print(f"  [METEO] Telechargement {date_min} → {date_max} ({pays})...")
        df_meteo = fetch_meteo(date_min, date_max, pays=pays)

        if df_meteo is not None:
            _save_cache(df_meteo)
        else:
            return _add_empty_meteo(df)

    # Merge avec les données de ventes
    df = df.merge(df_meteo, on="date", how="left")

    # Features dérivées
    df["temperature_max"] = df["temperature_max"].fillna(df["temperature_max"].median())
    df["temperature_min"] = df["temperature_min"].fillna(df["temperature_min"].median())
    df["precipitation_mm"] = df["precipitation_mm"].fillna(0)

    df["pluie_forte"] = (df["precipitation_mm"] > 10).astype(int)
    df["chaleur_extreme"] = (df["temperature_max"] > 38).astype(int)

    # Saison (Afrique de l'Ouest)
    df["saison"] = df["date"].dt.month.map(lambda m: _saison(m))
    df["est_hivernage"] = (df["saison"] == "hivernage").astype(int)

    return df


def _saison(mois):
    """Détermine la saison en Afrique de l'Ouest."""
    if mois in [6, 7, 8, 9, 10]:
        return "hivernage"
    elif mois in [11, 12, 1, 2]:
        return "saison_seche"
    else:
        return "transition"


def _load_cache(date_min, date_max):
    """Charge les données météo depuis le cache local."""
    if not METEO_CACHE.exists():
        return None

    df = pd.read_csv(METEO_CACHE, parse_dates=["date"])

    # Vérifier que le cache couvre la période demandée
    if df["date"].min().strftime("%Y-%m-%d") <= date_min and \
       df["date"].max().strftime("%Y-%m-%d") >= date_max:
        mask = (df["date"] >= date_min) & (df["date"] <= date_max)
        return df[mask].reset_index(drop=True)

    return None


def _save_cache(df_meteo):
    """Sauvegarde les données météo dans le cache."""
    METEO_CACHE.parent.mkdir(parents=True, exist_ok=True)

    if METEO_CACHE.exists():
        existing = pd.read_csv(METEO_CACHE, parse_dates=["date"])
        df_meteo = pd.concat([existing, df_meteo]).drop_duplicates(subset="date").sort_values("date")

    df_meteo.to_csv(METEO_CACHE, index=False)


def _add_empty_meteo(df):
    """Ajoute des colonnes météo vides si l'API est indisponible."""
    df["temperature_max"] = np.nan
    df["temperature_min"] = np.nan
    df["precipitation_mm"] = 0
    df["pluie_forte"] = 0
    df["chaleur_extreme"] = 0
    df["est_hivernage"] = df["date"].dt.month.isin([6, 7, 8, 9, 10]).astype(int)
    return df


def get_meteo_feature_columns():
    """Retourne la liste des features météo."""
    return [
        "temperature_max", "temperature_min", "precipitation_mm",
        "pluie_forte", "chaleur_extreme", "est_hivernage",
    ]
