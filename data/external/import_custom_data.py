"""
Importe des données de ventes depuis différents formats courants
et les convertit au format attendu par le pipeline.

Formats supportés :
    - Excel de caisse (colonnes variables)
    - Relevé Mobile Money (Orange Money, Wave, MTN MoMo)
    - CSV brut avec au minimum une colonne date et une colonne montant

Usage :
    python data/external/import_custom_data.py mon_fichier.xlsx
    python data/external/import_custom_data.py releve_wave.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


# Mapping des noms de colonnes courants vers notre format
COLONNES_DATE = [
    "date", "Date", "DATE", "jour", "Jour", "date_transaction",
    "transaction_date", "created_at", "created", "date_vente",
]

COLONNES_MONTANT = [
    "montant_total", "montant", "Montant", "MONTANT", "total", "Total",
    "chiffre_affaires", "ca", "CA", "sales", "Sales", "amount", "Amount",
    "revenue", "ventes", "Ventes", "prix_total", "net",
]

COLONNES_TRANSACTIONS = [
    "nb_transactions", "transactions", "nombre_transactions", "count",
    "nb_ventes", "quantity", "Quantity",
]


def detect_column(df, candidates):
    """Détecte une colonne parmi une liste de noms possibles."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def detect_date_column(df):
    """Détecte la colonne date, même si le nom n'est pas standard."""
    # D'abord chercher par nom
    col = detect_column(df, COLONNES_DATE)
    if col:
        return col

    # Sinon chercher la première colonne qui ressemble à des dates
    for col in df.columns:
        sample = df[col].dropna().head(10)
        try:
            pd.to_datetime(sample)
            return col
        except (ValueError, TypeError):
            continue

    return None


def detect_montant_column(df):
    """Détecte la colonne de montant."""
    col = detect_column(df, COLONNES_MONTANT)
    if col:
        return col

    # Chercher la première colonne numérique avec des grandes valeurs
    for col in df.columns:
        if df[col].dtype in ["float64", "int64", "float32", "int32"]:
            if df[col].mean() > 1000:
                return col

    return None


def load_file(filepath):
    """Charge un fichier (CSV, Excel, ou TSV)."""
    filepath = Path(filepath)

    if filepath.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    elif filepath.suffix == ".tsv":
        df = pd.read_csv(filepath, sep="\t")
    else:
        # Tenter CSV avec différents séparateurs
        try:
            df = pd.read_csv(filepath)
        except Exception:
            df = pd.read_csv(filepath, sep=";")

    return df


def convert_to_pipeline_format(df, use_case="supermarche"):
    """Convertit un DataFrame brut au format attendu par le pipeline."""

    # Détecter la colonne date
    date_col = detect_date_column(df)
    if date_col is None:
        raise ValueError(
            "Impossible de detecter la colonne de date.\n"
            f"Colonnes trouvees : {list(df.columns)}\n"
            "Renommez votre colonne de date en 'date'."
        )

    # Détecter la colonne montant
    montant_col = detect_montant_column(df)
    if montant_col is None:
        raise ValueError(
            "Impossible de detecter la colonne de montant.\n"
            f"Colonnes trouvees : {list(df.columns)}\n"
            "Renommez votre colonne de montant en 'montant_total'."
        )

    # Détecter la colonne transactions (optionnelle)
    trans_col = detect_column(df, COLONNES_TRANSACTIONS)

    print(f"  Colonne date     : '{date_col}'")
    print(f"  Colonne montant  : '{montant_col}'")
    print(f"  Colonne trans.   : '{trans_col or 'non trouvee (sera estimee)'}'")

    # Convertir les dates
    df["date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # Convertir le montant
    df["montant_total"] = pd.to_numeric(
        df[montant_col].astype(str).str.replace(r"[^\d.,\-]", "", regex=True).str.replace(",", "."),
        errors="coerce"
    )
    df = df.dropna(subset=["montant_total"])
    df["montant_total"] = df["montant_total"].astype(int)

    # Si les données sont par transaction, agréger par jour
    if len(df) > 0:
        nb_dates_uniques = df["date"].dt.date.nunique()
        ratio = len(df) / max(nb_dates_uniques, 1)

        if ratio > 2:
            print(f"  [INFO] Donnees transactionnelles detectees ({ratio:.0f} lignes/jour)")
            print(f"         Agregation par jour...")

            df_daily = df.groupby(df["date"].dt.date).agg(
                montant_total=("montant_total", "sum"),
                nb_transactions=("montant_total", "count"),
            ).reset_index()
            df_daily["date"] = pd.to_datetime(df_daily["date"])
        else:
            df_daily = df[["date", "montant_total"]].copy()
            if trans_col:
                df_daily["nb_transactions"] = df[trans_col]
            else:
                df_daily["nb_transactions"] = (df_daily["montant_total"] / 3500).astype(int).clip(lower=1)
    else:
        raise ValueError("Aucune donnee valide apres conversion.")

    # Colonnes finales
    if "nb_transactions" not in df_daily.columns:
        df_daily["nb_transactions"] = (df_daily["montant_total"] / 3500).astype(int).clip(lower=1)

    df_daily["nb_clients"] = (df_daily["nb_transactions"] * np.random.uniform(0.75, 0.95, len(df_daily))).astype(int)
    df_daily["use_case"] = use_case

    df_daily = df_daily.sort_values("date").reset_index(drop=True)

    # Supprimer les jours avec montant négatif ou nul
    df_daily = df_daily[df_daily["montant_total"] > 0]

    return df_daily[["date", "montant_total", "nb_transactions", "nb_clients", "use_case"]]


def main():
    parser = argparse.ArgumentParser(
        description="Importer des donnees de ventes vers le format pipeline"
    )
    parser.add_argument("fichier", help="Chemin vers le fichier (CSV, Excel, TSV)")
    parser.add_argument(
        "--use-case",
        choices=["supermarche", "restaurant", "mobile_money", "grossiste"],
        default="supermarche",
        help="Type de commerce (defaut: supermarche)"
    )
    parser.add_argument(
        "--output",
        help="Nom du fichier de sortie (defaut: ventes_custom.csv)"
    )

    args = parser.parse_args()
    filepath = Path(args.fichier)

    if not filepath.exists():
        print(f"[ERREUR] Fichier introuvable : {filepath}")
        sys.exit(1)

    print("=" * 50)
    print("IMPORT DE DONNEES PERSONNALISEES")
    print("=" * 50)
    print()
    print(f"  Fichier : {filepath.name}")
    print(f"  Format  : {filepath.suffix}")
    print(f"  Use case: {args.use_case}")
    print()

    # Charger
    print("[1/3] Chargement du fichier...")
    df = load_file(filepath)
    print(f"  {len(df)} lignes, {len(df.columns)} colonnes")
    print(f"  Colonnes : {list(df.columns)}")
    print()

    # Convertir
    print("[2/3] Conversion au format pipeline...")
    df_converted = convert_to_pipeline_format(df, use_case=args.use_case)
    print()

    # Sauvegarder
    output_name = args.output or f"ventes_custom_{args.use_case}.csv"
    output_path = DATA_RAW / output_name
    df_converted["date"] = df_converted["date"].dt.strftime("%Y-%m-%d")
    df_converted.to_csv(output_path, index=False)

    print(f"[3/3] Sauvegarde...")
    print(f"  [OK] {output_path}")
    print()
    print("=" * 50)
    print("RESUME")
    print("=" * 50)
    print(f"  Periode      : {df_converted['date'].min()} → {df_converted['date'].max()}")
    print(f"  Nb jours     : {len(df_converted)}")
    print(f"  CA moyen/jour: {df_converted['montant_total'].mean():,.0f} FCFA")
    print(f"  CA total     : {df_converted['montant_total'].sum():,.0f} FCFA")
    print()
    print("  Pour entrainer le modele avec ces donnees :")
    print(f"    python3 -c \"from src.model import train_pipeline; train_pipeline('{args.use_case}')\"")
    print()
    print("  Ou importez le CSV directement dans le dashboard Streamlit.")


if __name__ == "__main__":
    main()
