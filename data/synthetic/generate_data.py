"""
Génération de données synthétiques de ventes pour l'Afrique de l'Ouest.
Couvre 4 use cases : supermarché, restaurant, mobile money, grossiste.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


# Dates des fêtes majeures (source : calendriers officiels)
FETES = {
    "tabaski": [
        datetime(2022, 7, 9), datetime(2023, 6, 28), datetime(2024, 6, 16)
    ],
    "korite": [
        datetime(2022, 5, 2), datetime(2023, 4, 21), datetime(2024, 4, 10)
    ],
    "noel": [
        datetime(2022, 12, 25), datetime(2023, 12, 25), datetime(2024, 12, 25)
    ],
    "nouvel_an": [
        datetime(2022, 1, 1), datetime(2023, 1, 1), datetime(2024, 1, 1)
    ],
    "fete_nationale_sn": [
        datetime(2022, 4, 4), datetime(2023, 4, 4), datetime(2024, 4, 4)
    ],
    "fete_nationale_ci": [
        datetime(2022, 8, 7), datetime(2023, 8, 7), datetime(2024, 8, 7)
    ],
    "fete_travail": [
        datetime(2022, 5, 1), datetime(2023, 5, 1), datetime(2024, 5, 1)
    ],
    "mawlid": [
        datetime(2022, 10, 8), datetime(2023, 9, 27), datetime(2024, 9, 16)
    ],
}

# Périodes de Ramadan approximatives
RAMADAN_PERIODES = [
    (datetime(2022, 4, 2), datetime(2022, 5, 1)),
    (datetime(2023, 3, 23), datetime(2023, 4, 20)),
    (datetime(2024, 3, 11), datetime(2024, 4, 9)),
]

CONFIGS_USE_CASE = {
    "supermarche": {
        "base": 1_200_000,
        "bruit": 0.12,
        "croissance_annuelle": 0.08,
        "effet_weekend_sam": 1.35,
        "effet_weekend_dim": 1.15,
        "effet_debut_mois": 1.45,
        "effet_fin_mois": 1.20,
    },
    "restaurant": {
        "base": 350_000,
        "bruit": 0.18,
        "croissance_annuelle": 0.10,
        "effet_weekend_sam": 1.50,
        "effet_weekend_dim": 1.40,
        "effet_debut_mois": 1.25,
        "effet_fin_mois": 1.10,
    },
    "mobile_money": {
        "base": 180_000,
        "bruit": 0.22,
        "croissance_annuelle": 0.15,
        "effet_weekend_sam": 0.85,
        "effet_weekend_dim": 0.70,
        "effet_debut_mois": 1.60,
        "effet_fin_mois": 1.35,
    },
    "grossiste": {
        "base": 5_000_000,
        "bruit": 0.10,
        "croissance_annuelle": 0.06,
        "effet_weekend_sam": 0.90,
        "effet_weekend_dim": 0.40,
        "effet_debut_mois": 1.30,
        "effet_fin_mois": 1.15,
    },
}


def _est_pendant_ramadan(date):
    for debut, fin in RAMADAN_PERIODES:
        if debut.date() <= date.date() <= fin.date():
            return True
    return False


def _effet_fete(date):
    """Calcule le multiplicateur lié à la proximité d'une fête."""
    multiplicateur = 1.0
    for nom_fete, dates_fete in FETES.items():
        for d in dates_fete:
            diff = (date.date() - d.date()).days
            if diff == -1:
                multiplicateur = max(multiplicateur, 2.2)
            elif diff == -2:
                multiplicateur = max(multiplicateur, 1.8)
            elif diff == -3:
                multiplicateur = max(multiplicateur, 1.4)
            elif diff == 0:
                if nom_fete in ("tabaski", "korite"):
                    multiplicateur = max(multiplicateur, 0.6)
                else:
                    multiplicateur = max(multiplicateur, 1.3)
            elif diff == 1:
                multiplicateur = max(multiplicateur, 1.2)
    return multiplicateur


def generate_sales_data(
    start_date="2022-01-01",
    end_date="2024-12-31",
    use_case="supermarche",
    seed=42,
):
    """
    Génère un dataset de ventes réaliste.

    Intègre : saisonnalité hebdomadaire/mensuelle, effets fêtes,
    Ramadan, tendance de croissance, auto-corrélation entre jours.
    """
    np.random.seed(seed)
    cfg = CONFIGS_USE_CASE[use_case]

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    date_debut = pd.Timestamp(start_date)

    data = []
    montant_precedent = cfg["base"]

    for date in dates:
        montant = cfg["base"]

        # Tendance de croissance annuelle
        annees_ecoulees = (date - date_debut).days / 365.25
        montant *= (1 + cfg["croissance_annuelle"]) ** annees_ecoulees

        # Saisonnalité mensuelle (décembre = mois fort, février = mois creux)
        saisonnalite_mois = {
            1: 0.95, 2: 0.88, 3: 0.92, 4: 0.95, 5: 0.97, 6: 1.0,
            7: 0.93, 8: 0.90, 9: 1.05, 10: 1.02, 11: 1.05, 12: 1.20
        }
        montant *= saisonnalite_mois.get(date.month, 1.0)

        # Effet jour de la semaine
        if date.weekday() == 5:
            montant *= cfg["effet_weekend_sam"]
        elif date.weekday() == 6:
            montant *= cfg["effet_weekend_dim"]
        elif date.weekday() == 0:
            montant *= 0.92

        # Effet début de mois (jours 1-5 : paie)
        if date.day <= 5:
            montant *= cfg["effet_debut_mois"]

        # Effet fin de mois (jours 26-31)
        if date.day >= 26:
            montant *= cfg["effet_fin_mois"]

        # Effet fêtes
        montant *= _effet_fete(date)

        # Effet Ramadan
        if _est_pendant_ramadan(date):
            if use_case == "restaurant":
                montant *= 0.70
            elif use_case == "supermarche":
                montant *= 1.15
            elif use_case == "mobile_money":
                montant *= 1.10

        # Auto-corrélation avec le jour précédent (inertie)
        montant = 0.85 * montant + 0.15 * montant_precedent

        # Bruit aléatoire
        montant *= (1 + np.random.normal(0, cfg["bruit"]))
        montant = max(montant * 0.3, montant)

        montant_precedent = montant

        nb_transactions = max(1, int(montant / np.random.uniform(3000, 4500)))
        nb_clients = max(1, int(nb_transactions * np.random.uniform(0.75, 0.95)))

        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "montant_total": round(montant),
            "nb_transactions": nb_transactions,
            "nb_clients": nb_clients,
            "use_case": use_case,
        })

    return pd.DataFrame(data)


def generate_calendrier_evenements():
    """Génère le fichier calendrier_evenements.csv."""
    evenements = []

    impacts = {
        "tabaski": ("fete_religieuse", "tres_fort"),
        "korite": ("fete_religieuse", "tres_fort"),
        "noel": ("fete_religieuse", "fort"),
        "nouvel_an": ("fete_civile", "fort"),
        "fete_nationale_sn": ("fete_nationale", "moyen"),
        "fete_nationale_ci": ("fete_nationale", "moyen"),
        "fete_travail": ("fete_civile", "faible"),
        "mawlid": ("fete_religieuse", "moyen"),
    }

    for nom, dates_list in FETES.items():
        type_evt, impact = impacts[nom]
        pays = "CI" if "ci" in nom else "SN"
        for d in dates_list:
            evenements.append({
                "date": d.strftime("%Y-%m-%d"),
                "evenement": nom,
                "type": type_evt,
                "impact_estime": impact,
                "pays": pays,
            })

    # Ajouter les périodes de Ramadan
    for debut, fin in RAMADAN_PERIODES:
        evenements.append({
            "date": debut.strftime("%Y-%m-%d"),
            "evenement": "debut_ramadan",
            "type": "fete_religieuse",
            "impact_estime": "fort",
            "pays": "SN",
        })

    return pd.DataFrame(evenements).sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    for uc in ["supermarche", "restaurant", "mobile_money", "grossiste"]:
        df = generate_sales_data(use_case=uc)
        filepath = output_dir / f"ventes_{uc}.csv"
        df.to_csv(filepath, index=False)
        print(f"[OK] {filepath.name} — {len(df)} lignes")

    df_evt = generate_calendrier_evenements()
    filepath_evt = output_dir / "calendrier_evenements.csv"
    df_evt.to_csv(filepath_evt, index=False)
    print(f"[OK] {filepath_evt.name} — {len(df_evt)} evenements")
