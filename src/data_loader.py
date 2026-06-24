"""Chargement et validation des données."""

import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO, StringIO

from src.utils import PROJECT_ROOT

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def _try_read_file(filepath):
    """Lit un CSV, Excel ou JSON depuis un path ou un file-like (Streamlit UploadedFile)."""
    # Déterminer l'extension
    name = ""
    if hasattr(filepath, "name"):
        name = filepath.name.lower()
    elif isinstance(filepath, (str, Path)):
        name = str(filepath).lower()

    if hasattr(filepath, "read"):
        filepath.seek(0)

    if name.endswith(".json"):
        df = pd.read_json(filepath)
        # JSON peut être un tableau d'objets ou un dict de colonnes — pandas gère les deux
        return df
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(filepath)
    else:
        # CSV avec détection du séparateur
        if hasattr(filepath, "read"):
            filepath.seek(0)
            sample = filepath.read(2048)
            filepath.seek(0)
            if isinstance(sample, bytes):
                sample = sample.decode("utf-8", errors="ignore")
            sep = ";" if sample.count(";") > sample.count(",") else ","
            return pd.read_csv(filepath, sep=sep)
        return pd.read_csv(filepath)


def _find_column(df, candidates):
    """Trouve une colonne parmi une liste de noms possibles (insensible à la casse)."""
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in df_cols_lower:
            return df_cols_lower[candidate.lower()]
    return None


def _detect_date_column(df):
    """Détecte la colonne date par nom ou par contenu."""
    noms_date = ["date", "jour", "date_transaction", "transaction_date", "created_at", "ds", "day"]
    col = _find_column(df, noms_date)
    if col:
        return col

    for col in df.columns:
        sample = df[col].dropna().head(20).astype(str)
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() > 15:
                return col
        except Exception:
            continue
    return None


def _detect_montant_column(df):
    """Détecte la colonne de montant/ventes."""
    noms_montant = [
        "montant_total", "montant", "total", "sales", "amount", "revenue",
        "ventes", "ca", "chiffre_affaires", "net", "price", "value",
    ]
    col = _find_column(df, noms_montant)
    if col:
        return col

    for col in df.columns:
        if df[col].dtype in ["float64", "int64", "float32", "int32"]:
            if df[col].mean() > 0 and col.lower() not in ["id", "store_nbr", "onpromotion"]:
                return col
    return None


def _detect_transactions_column(df):
    """Détecte la colonne du nombre de transactions."""
    noms = ["nb_transactions", "transactions", "count", "quantity", "nb_ventes", "num_transactions"]
    return _find_column(df, noms)


def _smart_convert(df):
    """
    Convertit intelligemment n'importe quel CSV de ventes au format pipeline.
    Gère : colonnes nommées différemment, données transactionnelles, formats exotiques.
    """
    # Détecter les colonnes
    date_col = _detect_date_column(df)
    montant_col = _detect_montant_column(df)
    trans_col = _detect_transactions_column(df)

    if date_col is None:
        raise ValueError(
            f"Impossible de detecter la colonne de date. "
            f"Colonnes trouvees : {list(df.columns)}. "
            f"Renommez votre colonne de date en 'date'."
        )

    if montant_col is None:
        raise ValueError(
            f"Impossible de detecter la colonne de montant/ventes. "
            f"Colonnes trouvees : {list(df.columns)}. "
            f"Renommez votre colonne en 'montant_total' ou 'sales'."
        )

    # Renommer
    rename_map = {date_col: "date", montant_col: "montant_total"}
    if trans_col:
        rename_map[trans_col] = "nb_transactions"
    df = df.rename(columns=rename_map)

    # Convertir la date
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # Convertir le montant (gérer les formats avec espaces, virgules, symboles)
    if df["montant_total"].dtype == object:
        df["montant_total"] = (
            df["montant_total"].astype(str)
            .str.replace(r"[^\d.,\-]", "", regex=True)
            .str.replace(",", ".")
        )
    df["montant_total"] = pd.to_numeric(df["montant_total"], errors="coerce")
    df = df.dropna(subset=["montant_total"])

    # Supprimer les ventes nulles ou négatives
    df = df[df["montant_total"] > 0]

    if df.empty:
        raise ValueError("Aucune donnee valide apres conversion (montants tous nuls ou negatifs).")

    # Si données transactionnelles (plusieurs lignes par jour), agréger
    nb_dates_uniques = df["date"].dt.date.nunique()
    ratio = len(df) / max(nb_dates_uniques, 1)

    if ratio > 1.5:
        df = df.groupby(df["date"].dt.date).agg(
            montant_total=("montant_total", "sum"),
            nb_transactions=("montant_total", "count"),
        ).reset_index()
        df["date"] = pd.to_datetime(df["date"])
    else:
        if "nb_transactions" not in df.columns:
            df["nb_transactions"] = (df["montant_total"] / 3500).clip(lower=1).astype(int)
        df = df[["date", "montant_total", "nb_transactions"]]

    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset="date", keep="last")

    return df


def load_ventes(use_case="supermarche", filepath=None):
    """
    Charge les données de ventes depuis n'importe quel format.
    Détecte automatiquement les colonnes et agrège si nécessaire.
    """
    if filepath is None:
        filepath = DATA_RAW / f"ventes_{use_case}.csv"

    df = _try_read_file(filepath)
    df = _smart_convert(df)

    return df


def load_evenements():
    """Charge le calendrier des événements."""
    filepath = DATA_RAW / "calendrier_evenements.csv"
    if not filepath.exists():
        return pd.DataFrame(columns=["date", "evenement", "type", "impact_estime", "pays"])

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    return df
