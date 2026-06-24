"""
Télécharge et adapte le dataset "Store Sales - Time Series Forecasting" (Equateur)
depuis Kaggle pour entraîner le modèle avec des données réelles.

Prérequis :
    pip install kaggle
    Placer votre fichier kaggle.json dans ~/.kaggle/
    (Obtenir la clé API sur https://www.kaggle.com/settings → "Create New Token")

Usage :
    python data/external/download_kaggle_data.py
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import zipfile
import shutil


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
DOWNLOAD_DIR = DATA_EXTERNAL / "kaggle_store_sales"


def setup_kaggle_credentials():
    """Cherche kaggle.json dans le projet, sinon utilise ~/.kaggle/."""
    import os
    local_kaggle = PROJECT_ROOT / "kaggle.json"
    if local_kaggle.exists():
        os.environ["KAGGLE_CONFIG_DIR"] = str(PROJECT_ROOT)
        print(f"  [OK] Cle API trouvee : {local_kaggle}")
    else:
        print("  [INFO] kaggle.json non trouve dans le projet, utilisation de ~/.kaggle/")


def install_kaggle():
    """Installe le package kaggle si absent."""
    try:
        import kaggle
    except ImportError:
        print("[INFO] Installation du package kaggle...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])


def download_dataset():
    """Télécharge le dataset depuis Kaggle."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Telechargement du dataset Kaggle 'Store Sales - Time Series'...")
    print("      (Equateur — 54 magasins, 33 familles de produits, 2013-2017)")
    print()

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "kaggle", "competitions", "download",
                "-c", "store-sales-time-series-forecasting",
                "-p", str(DOWNLOAD_DIR),
            ],
            capture_output=True, text=True,
        )

        if result.returncode != 0:
            if "403" in result.stderr or "Could not find" in result.stderr:
                print("[ERREUR] Vous devez accepter les regles de la competition sur Kaggle :")
                print("         https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules")
                print()
                print("         Puis relancez ce script.")
                return False
            elif "kaggle.json" in result.stderr or "Could not find kaggle" in result.stderr:
                print("[ERREUR] Configuration Kaggle manquante.")
                print()
                print("  1. Allez sur https://www.kaggle.com/settings")
                print("  2. Cliquez 'Create New Token' → telecharge kaggle.json")
                print("  3. Placez-le dans ~/.kaggle/kaggle.json")
                print("  4. chmod 600 ~/.kaggle/kaggle.json")
                print()
                print("  Puis relancez ce script.")
                return False
            else:
                print(f"[ERREUR] {result.stderr}")
                return False

    except FileNotFoundError:
        print("[ERREUR] Kaggle CLI introuvable. Installez avec : pip install kaggle")
        return False

    # Dézipper
    zip_files = list(DOWNLOAD_DIR.glob("*.zip"))
    for zf in zip_files:
        print(f"  Extraction de {zf.name}...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(DOWNLOAD_DIR)
        zf.unlink()

    print("  [OK] Telechargement termine.")
    return True


def adapt_to_pipeline():
    """
    Transforme les données Kaggle au format attendu par notre pipeline.
    Agrège par jour pour simuler un commerce unique (ou par magasin).
    """
    print("\n[2/4] Chargement des donnees brutes...")

    train_path = DOWNLOAD_DIR / "train.csv"
    if not train_path.exists():
        print(f"[ERREUR] Fichier {train_path} introuvable.")
        print("         Verifiez que le telechargement a fonctionne.")
        return None

    df = pd.read_csv(train_path, parse_dates=["date"])
    print(f"  {len(df):,} lignes chargees (54 magasins x 33 familles x 1684 jours)")

    # Charger les jours fériés
    holidays_path = DOWNLOAD_DIR / "holidays_events.csv"
    df_holidays = None
    if holidays_path.exists():
        df_holidays = pd.read_csv(holidays_path, parse_dates=["date"])
        print(f"  {len(df_holidays)} evenements/fetes charges")

    print("\n[3/4] Adaptation au format du pipeline...")

    # --- Option A : Agrégation totale (tous les magasins = un seul commerce) ---
    df_daily = df.groupby("date").agg(
        montant_total=("sales", "sum"),
        nb_transactions=("sales", "count"),
    ).reset_index()

    # Nombre de clients estimé (pas dans le dataset, on approxime)
    df_daily["nb_clients"] = (df_daily["nb_transactions"] * 0.85).astype(int)
    df_daily["use_case"] = "supermarche"

    # Convertir les montants en FCFA (1 USD ~ 600 FCFA)
    # Les ventes Equateur sont en USD, on multiplie pour avoir un ordre de grandeur réaliste
    taux_conversion = 600
    df_daily["montant_total"] = (df_daily["montant_total"] * taux_conversion).round(0).astype(int)

    # --- Option B : Par magasin (top 5 magasins comme exemples) ---
    df_par_magasin = df.groupby(["date", "store_nbr"]).agg(
        montant_total=("sales", "sum"),
        nb_transactions=("sales", "count"),
    ).reset_index()

    # Top 5 magasins par volume
    top_magasins = df.groupby("store_nbr")["sales"].sum().nlargest(5).index.tolist()

    # --- Sauvegarde ---
    print("\n[4/4] Sauvegarde des fichiers adaptes...")

    # Fichier principal agrégé
    output_path = DATA_RAW / "ventes_equateur_agregees.csv"
    df_daily.to_csv(output_path, index=False)
    print(f"  [OK] {output_path.name} — {len(df_daily)} jours (tous magasins agreges)")

    # Fichiers par magasin
    for store_id in top_magasins:
        df_store = df_par_magasin[df_par_magasin["store_nbr"] == store_id].copy()
        df_store["montant_total"] = (df_store["montant_total"] * taux_conversion).round(0).astype(int)
        df_store["nb_clients"] = (df_store["nb_transactions"] * 0.85).astype(int)
        df_store["use_case"] = "supermarche"
        df_store = df_store.drop(columns=["store_nbr"])

        output_path = DATA_RAW / f"ventes_equateur_magasin_{store_id}.csv"
        df_store.to_csv(output_path, index=False)
        print(f"  [OK] {output_path.name} — {len(df_store)} jours")

    # Fichier événements adapté
    if df_holidays is not None:
        df_evt = df_holidays[df_holidays["transferred"] == False].copy()
        df_evt = df_evt[["date", "description", "type"]].rename(columns={
            "description": "evenement",
        })
        df_evt["impact_estime"] = df_evt["type"].map({
            "Holiday": "fort",
            "Transfer": "moyen",
            "Event": "moyen",
            "Bridge": "faible",
            "Work Day": "faible",
        }).fillna("moyen")
        df_evt["pays"] = "EC"
        df_evt["date"] = df_evt["date"].dt.strftime("%Y-%m-%d")

        evt_path = DATA_RAW / "calendrier_evenements_equateur.csv"
        df_evt.to_csv(evt_path, index=False)
        print(f"  [OK] {evt_path.name} — {len(df_evt)} evenements")

    return df_daily


def show_stats(df):
    """Affiche les statistiques du dataset adapté."""
    if df is None:
        return

    print("\n" + "=" * 50)
    print("STATISTIQUES DU DATASET ADAPTE")
    print("=" * 50)
    print(f"  Periode      : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Nb jours     : {len(df)}")
    print(f"  CA moyen/jour: {df['montant_total'].mean():,.0f} FCFA")
    print(f"  CA min       : {df['montant_total'].min():,.0f} FCFA")
    print(f"  CA max       : {df['montant_total'].max():,.0f} FCFA")
    print()
    print("  Pour entrainer avec ces donnees :")
    print("  >>> from src.model import train_pipeline")
    print("  >>> train_pipeline('supermarche')  # pointera vers le fichier Equateur")
    print()
    print("  Ou via le dashboard : importez le CSV via la barre laterale.")


def main():
    install_kaggle()
    setup_kaggle_credentials()

    print("=" * 50)
    print("IMPORT DATASET KAGGLE → FORMAT PIPELINE")
    print("=" * 50)
    print()

    success = download_dataset()
    if not success:
        # Proposer une alternative manuelle
        print()
        print("-" * 50)
        print("ALTERNATIVE MANUELLE :")
        print("-" * 50)
        print()
        print("  1. Allez sur https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data")
        print("  2. Telechargez 'train.csv' et 'holidays_events.csv'")
        print(f"  3. Placez-les dans : {DOWNLOAD_DIR}/")
        print("  4. Relancez ce script")
        return

    df = adapt_to_pipeline()
    show_stats(df)

    # Nettoyage du dossier temporaire (garder seulement les CSV utiles)
    print("\n[INFO] Les fichiers bruts Kaggle sont dans :")
    print(f"       {DOWNLOAD_DIR}/")
    print("       Vous pouvez les supprimer apres verification.")


if __name__ == "__main__":
    main()
