"""
Script d'entrainement automatique sur toutes les donnees disponibles.
Detecte les fichiers dans data/raw/, entraine un modele par fichier,
selectionne le meilleur et le sauvegarde comme modele principal.

Usage :
    python3 train_all.py
    python3 train_all.py --best-only       # Entraine uniquement le meilleur dataset
    python3 train_all.py --file ventes_equateur_magasin_3.csv
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_ventes
from src.feature_engineering import build_features
from src.model import train_xgboost, MODELS_DIR


DATA_RAW = Path(__file__).parent / "data" / "raw"
RESULTS_FILE = Path(__file__).parent / "resultats_entrainement.json"


def find_datasets():
    """Trouve tous les fichiers de ventes dans data/raw/."""
    patterns = ["*.csv", "*.xlsx", "*.json"]
    fichiers = []
    for pattern in patterns:
        for f in DATA_RAW.glob(pattern):
            if "calendrier" not in f.name and "evenement" not in f.name:
                fichiers.append(f)
    return sorted(fichiers)


def train_single(filepath, save_model=False):
    """Entraine sur un seul fichier. Retourne les metriques."""
    name = filepath.stem
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    try:
        df = load_ventes(filepath=str(filepath))
        print(f"  Donnees   : {len(df)} jours ({df['date'].min().date()} -> {df['date'].max().date()})")
        print(f"  CA moyen  : {df['montant_total'].mean():,.0f}")

        df = build_features(df)
        print(f"  Features  : {len(df)} observations apres feature engineering")

        if len(df) < 30:
            print(f"  [SKIP] Pas assez de donnees ({len(df)} < 30)")
            return None

        model, metrics, importance = train_xgboost(df, save=save_model)

        print(f"  MAPE      : {metrics['mape']:.1f}%")
        print(f"  MAE       : {metrics['mae']:,.0f}")
        print(f"  RMSE      : {metrics['rmse']:,.0f}")
        if metrics.get("recall_pics"):
            print(f"  Recall    : {metrics['recall_pics']:.1f}%")
        print(f"  CV MAPE   : {metrics['cv_mape_mean']:.2f}% (+/- {metrics['cv_mape_std']:.2f}%)")
        print(f"  Top features : {', '.join(importance.head(3)['feature'].tolist())}")

        return {
            "fichier": filepath.name,
            "nb_jours": len(df),
            "periode": f"{df['date'].min().date()} -> {df['date'].max().date()}",
            "ca_moyen": round(float(df["montant_total"].mean())),
            "mae": round(float(metrics["mae"])),
            "mape": round(float(metrics["mape"]), 2),
            "rmse": round(float(metrics["rmse"])),
            "recall_pics": round(float(metrics.get("recall_pics", 0) or 0), 1),
            "cv_mape": round(float(metrics["cv_mape_mean"]), 2),
            "top_features": importance.head(5)["feature"].tolist(),
        }

    except Exception as e:
        print(f"  [ERREUR] {e}")
        return {"fichier": filepath.name, "erreur": str(e)}


def train_all(save_best=True):
    """Entraine sur tous les datasets et sauvegarde le meilleur modele."""
    fichiers = find_datasets()

    if not fichiers:
        print("[ERREUR] Aucun fichier de donnees trouve dans data/raw/")
        print("         Lancez d'abord : python3 data/synthetic/generate_data.py")
        return

    print(f"\n{'#'*60}")
    print(f"  ENTRAINEMENT AUTOMATIQUE — {len(fichiers)} datasets detectes")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    results = {}
    best_name = None
    best_mape = float("inf")

    for filepath in fichiers:
        result = train_single(filepath, save_model=False)
        if result and "erreur" not in result:
            results[filepath.stem] = result
            if result["mape"] < best_mape:
                best_mape = result["mape"]
                best_name = filepath

    # Sauvegarder les resultats
    results["_meta"] = {
        "date_entrainement": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nb_datasets": len(results) - 1,
        "meilleur_dataset": best_name.name if best_name else None,
        "meilleur_mape": best_mape if best_mape < float("inf") else None,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'#'*60}")
    print(f"  RESUME")
    print(f"{'#'*60}")
    print(f"\n  Resultats sauvegardes : {RESULTS_FILE}")

    if best_name:
        print(f"\n  Meilleur dataset : {best_name.name}")
        print(f"  Meilleur MAPE    : {best_mape:.1f}%")

        if save_best:
            print(f"\n  Entrainement du modele final sur {best_name.name}...")
            train_single(best_name, save_model=True)
            print(f"\n  [OK] Modele sauvegarde dans {MODELS_DIR / 'xgboost_model.pkl'}")
    else:
        print("\n  [ERREUR] Aucun modele n'a pu etre entraine.")

    # Tableau recapitulatif
    print(f"\n\n{'='*60}")
    print(f"  {'Dataset':<35} {'Jours':<8} {'MAPE':<8} {'MAE':<12}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*12}")
    for name, r in sorted(results.items(), key=lambda x: x[1].get("mape", 999)):
        if name == "_meta" or "erreur" in r:
            continue
        marker = " <-- BEST" if r["mape"] == best_mape else ""
        print(f"  {name:<35} {r['nb_jours']:<8} {r['mape']:<8.1f} {r['mae']:<12,}{marker}")
    print(f"  {'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Entrainement automatique sur tous les datasets")
    parser.add_argument("--file", help="Entrainer uniquement sur ce fichier")
    parser.add_argument("--best-only", action="store_true", help="Garder uniquement le meilleur modele")
    args = parser.parse_args()

    if args.file:
        filepath = DATA_RAW / args.file
        if not filepath.exists():
            print(f"[ERREUR] Fichier introuvable : {filepath}")
            sys.exit(1)
        train_single(filepath, save_model=True)
    else:
        train_all(save_best=True)


if __name__ == "__main__":
    main()
