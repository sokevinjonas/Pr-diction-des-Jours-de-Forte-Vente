"""Chargement et validation des données — accepte n'importe quel format."""

import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO, StringIO

from src.utils import PROJECT_ROOT

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def _try_read_file(filepath):
    """Lit un CSV, Excel ou JSON depuis un path ou un file-like."""
    name = ""
    if hasattr(filepath, "name"):
        name = filepath.name.lower()
    elif isinstance(filepath, (str, Path)):
        name = str(filepath).lower()

    if hasattr(filepath, "seek"):
        filepath.seek(0)

    if name.endswith(".json"):
        return pd.read_json(filepath)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(filepath)
    else:
        if hasattr(filepath, "read"):
            filepath.seek(0)
            sample = filepath.read(4096)
            filepath.seek(0)
            if isinstance(sample, bytes):
                sample = sample.decode("utf-8", errors="ignore")
            sep = ";" if sample.count(";") > sample.count(",") else ","
            return pd.read_csv(filepath, sep=sep)
        return pd.read_csv(filepath)


def _find_column(columns, candidates):
    """Trouve une colonne parmi des candidats (insensible à la casse et aux espaces)."""
    cols_normalized = {c.lower().strip().replace(" ", "_"): c for c in columns}
    for candidate in candidates:
        normalized = candidate.lower().strip().replace(" ", "_")
        if normalized in cols_normalized:
            return cols_normalized[normalized]
    return None


def _detect_date_column(df):
    """Détecte la colonne qui contient des dates."""
    candidates = ["date", "jour", "ds", "day", "date_transaction", "transaction_date", "created_at", "fecha"]
    col = _find_column(df.columns, candidates)
    if col:
        return col

    for col in df.columns:
        if df[col].dtype == "object" or "date" in str(df[col].dtype).lower():
            sample = df[col].dropna().head(20)
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() >= 15:
                    return col
            except Exception:
                continue
    return None


def _detect_montant_column(df):
    """Détecte la colonne qui contient les montants/ventes."""
    candidates = [
        "montant_total", "montant", "total", "sales", "amount", "revenue",
        "ventes", "ca", "CA", "chiffre_affaires", "net", "value", "income",
        "total_sales", "daily_sales", "turnover",
    ]
    col = _find_column(df.columns, candidates)
    if col:
        return col

    # Fallback : première colonne numérique qui n'est pas un ID ou un compteur
    ignore = {"id", "store_nbr", "store_id", "onpromotion", "promo", "index"}
    for col in df.columns:
        if col.lower().strip() in ignore:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].mean() > 0:
                return col
    return None


def _detect_transactions_column(df):
    """Détecte la colonne du nombre de transactions."""
    candidates = ["nb_transactions", "transactions", "count", "quantity", "nb_ventes", "num_transactions"]
    return _find_column(df.columns, candidates)


def _smart_convert(df):
    """
    Convertit intelligemment n'importe quel DataFrame de ventes au format pipeline.
    """
    original_columns = list(df.columns)

    # --- Étape 1 : Détecter les colonnes ---
    date_col = _detect_date_column(df)
    montant_col = _detect_montant_column(df)
    trans_col = _detect_transactions_column(df)

    if date_col is None:
        raise ValueError(
            f"Impossible de detecter la colonne de date. "
            f"Colonnes trouvees : {original_columns}. "
            f"Renommez votre colonne de date en 'date'."
        )

    if montant_col is None:
        raise ValueError(
            f"Impossible de detecter la colonne de montant/ventes. "
            f"Colonnes trouvees : {original_columns}. "
            f"Renommez votre colonne en 'montant_total' ou 'sales'."
        )

    # --- Étape 2 : Extraire et renommer les colonnes utiles ---
    result = pd.DataFrame()
    result["date"] = df[date_col]
    result["montant_total"] = df[montant_col]
    if trans_col:
        result["nb_transactions"] = df[trans_col]

    # --- Étape 3 : Convertir la date ---
    result["date"] = pd.to_datetime(result["date"], dayfirst=True, errors="coerce")
    result = result.dropna(subset=["date"])

    # --- Étape 4 : Convertir le montant ---
    if result["montant_total"].dtype == object:
        result["montant_total"] = (
            result["montant_total"].astype(str)
            .str.replace(r"[^\d.,\-]", "", regex=True)
            .str.replace(",", ".")
        )
    result["montant_total"] = pd.to_numeric(result["montant_total"], errors="coerce")
    result = result.dropna(subset=["montant_total"])

    # Supprimer les ventes nulles ou négatives
    result = result[result["montant_total"] > 0]

    if result.empty:
        raise ValueError("Aucune donnee valide apres conversion (montants tous nuls ou negatifs).")

    # --- Étape 5 : Agréger si données transactionnelles ---
    nb_dates_uniques = result["date"].dt.date.nunique()
    ratio = len(result) / max(nb_dates_uniques, 1)

    if ratio > 1.5:
        result = result.groupby(result["date"].dt.date).agg(
            montant_total=("montant_total", "sum"),
            nb_transactions=("montant_total", "count"),
        ).reset_index()
        result["date"] = pd.to_datetime(result["date"])
    else:
        if "nb_transactions" not in result.columns:
            result["nb_transactions"] = (result["montant_total"] / 3500).clip(lower=1).astype(int)
        result = result[["date", "montant_total", "nb_transactions"]]

    result = result.sort_values("date").reset_index(drop=True)
    result = result.drop_duplicates(subset="date", keep="last")

    return result


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
