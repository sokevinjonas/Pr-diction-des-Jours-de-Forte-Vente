"""Chargement et validation des données."""

import pandas as pd
from pathlib import Path

from src.utils import PROJECT_ROOT

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


COLONNES_REQUISES_VENTES = {"date", "montant_total", "nb_transactions"}


def load_ventes(use_case="supermarche", filepath=None):
    """
    Charge les données de ventes.
    Accepte un filepath custom (pour upload utilisateur) ou charge le fichier par défaut.
    """
    if filepath is None:
        filepath = DATA_RAW / f"ventes_{use_case}.csv"

    df = pd.read_csv(filepath)

    colonnes_manquantes = COLONNES_REQUISES_VENTES - set(df.columns)
    if colonnes_manquantes:
        raise ValueError(
            f"Colonnes manquantes dans le CSV : {colonnes_manquantes}. "
            f"Colonnes attendues : {COLONNES_REQUISES_VENTES}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Supprimer les doublons de dates
    df = df.drop_duplicates(subset="date", keep="last")

    return df


def load_evenements():
    """Charge le calendrier des événements."""
    filepath = DATA_RAW / "calendrier_evenements.csv"
    if not filepath.exists():
        return pd.DataFrame(columns=["date", "evenement", "type", "impact_estime", "pays"])

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    return df
