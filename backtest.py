"""
Backtesting — Valider le modele en predisant le passe connu.

Principe :
    On a 1 an de donnees (ex: jan 2017 → dec 2017).
    On entraine sur les 6 premiers mois (jan → juin).
    On predit semaine par semaine les 6 mois restants (juil → dec).
    On compare chaque prediction avec la realite.

Cela permet de PROUVER que le modele fonctionne avant de l'utiliser sur le vrai futur.

Usage :
    python3 backtest.py
    python3 backtest.py --file data/raw/ventes_equateur_magasin_3.csv
    python3 backtest.py --train-ratio 0.5 --step 7
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from src.data_loader import load_ventes
from src.feature_engineering import build_features, get_feature_columns
from src.model import MODELS_DIR

import xgboost as xgb


RESULTS_DIR = Path(__file__).parent / "resultats"
RESULTS_DIR.mkdir(exist_ok=True)


def backtest(
    filepath=None,
    use_case="supermarche",
    train_ratio=0.5,
    step_days=7,
    retrain_every=30,
):
    """
    Backtest walk-forward :
    1. Entraine sur les premiers X% des donnees
    2. Predit les `step_days` jours suivants
    3. Compare avec la realite
    4. Avance de `step_days` jours
    5. Re-entraine tous les `retrain_every` jours (simule un usage reel)
    """

    # Charger toutes les donnees
    if filepath:
        df_full = load_ventes(filepath=filepath)
    else:
        df_full = load_ventes(use_case=use_case)

    n_total = len(df_full)
    split_idx = int(n_total * train_ratio)

    if split_idx < 30:
        raise ValueError(f"Pas assez de donnees pour l'entrainement ({split_idx} jours)")

    date_split = df_full.iloc[split_idx]["date"]
    date_fin = df_full["date"].max()

    print(f"\n{'#'*60}")
    print(f"  BACKTESTING — Prediction du passe connu")
    print(f"{'#'*60}")
    print(f"\n  Fichier        : {filepath or use_case}")
    print(f"  Total          : {n_total} jours")
    print(f"  Entrainement   : {split_idx} jours ({df_full['date'].min().date()} → {date_split.date()})")
    print(f"  A predire      : {n_total - split_idx} jours ({date_split.date()} → {date_fin.date()})")
    print(f"  Pas            : {step_days} jours")
    print(f"  Re-entrainement: tous les {retrain_every} jours")
    print()

    # Variables pour le walk-forward
    feature_cols = get_feature_columns()
    all_predictions = []
    all_actuals = []
    all_dates = []
    weekly_results = []

    cursor = split_idx
    last_train_cursor = 0
    model = None
    n_retrains = 0

    while cursor < n_total:
        # Re-entrainer si necessaire
        if model is None or (cursor - last_train_cursor) >= retrain_every:
            df_train = df_full.iloc[:cursor].copy()
            df_train_feat = build_features(df_train)

            features_dispo = [c for c in feature_cols if c in df_train_feat.columns]
            X_train = df_train_feat[features_dispo]
            y_train = df_train_feat["montant_total"]

            model = xgb.XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train, verbose=False)
            last_train_cursor = cursor
            n_retrains += 1

        # Predire les prochains `step_days` jours
        end_cursor = min(cursor + step_days, n_total)
        df_known = df_full.iloc[:cursor].copy()

        step_preds = []
        step_actuals = []
        step_dates = []

        for i in range(cursor, end_cursor):
            actual = df_full.iloc[i]["montant_total"]
            date_pred = df_full.iloc[i]["date"]

            # Construire les features pour ce jour
            row = pd.DataFrame({"date": [date_pred], "montant_total": [np.nan]})
            df_temp = pd.concat([df_known, row], ignore_index=True)
            df_temp_feat = build_features(df_temp)

            if df_temp_feat.empty:
                continue

            last_row = df_temp_feat.iloc[[-1]]
            features_row = [c for c in features_dispo if c in last_row.columns]
            X_pred = last_row[features_row]

            pred = float(model.predict(X_pred)[0])
            pred = max(0, pred)

            step_preds.append(pred)
            step_actuals.append(actual)
            step_dates.append(date_pred)

            # Ajouter la prediction a l'historique pour les lags suivants
            df_known = pd.concat([
                df_known,
                pd.DataFrame({"date": [date_pred], "montant_total": [pred]})
            ], ignore_index=True)

        if step_preds:
            # Metriques de ce pas
            mae_step = mean_absolute_error(step_actuals, step_preds)
            mape_step = mean_absolute_percentage_error(step_actuals, step_preds) * 100

            week_start = step_dates[0].strftime("%Y-%m-%d")
            week_end = step_dates[-1].strftime("%Y-%m-%d")

            weekly_results.append({
                "periode": f"{week_start} → {week_end}",
                "nb_jours": len(step_preds),
                "mae": round(mae_step),
                "mape": round(mape_step, 1),
                "prevu_moyen": round(np.mean(step_preds)),
                "reel_moyen": round(np.mean(step_actuals)),
            })

            all_predictions.extend(step_preds)
            all_actuals.extend(step_actuals)
            all_dates.extend(step_dates)

            status = "OK" if mape_step < 20 else "MOYEN" if mape_step < 30 else "FAIBLE"
            print(f"  [{status:6}] {week_start} → {week_end} | MAPE: {mape_step:5.1f}% | MAE: {mae_step:>12,.0f}")

        cursor = end_cursor

    # Resultats globaux
    if not all_predictions:
        print("\n  [ERREUR] Aucune prediction reussie.")
        return None

    global_mae = mean_absolute_error(all_actuals, all_predictions)
    global_mape = mean_absolute_percentage_error(all_actuals, all_predictions) * 100
    global_rmse = np.sqrt(np.mean((np.array(all_actuals) - np.array(all_predictions))**2))

    # Detection des pics
    actuals_arr = np.array(all_actuals)
    preds_arr = np.array(all_predictions)
    moy_globale = actuals_arr.mean()

    vrais_pics = actuals_arr > moy_globale * 1.3
    preds_pics = preds_arr > moy_globale * 1.3

    recall_pics = 0
    precision_pics = 0
    if vrais_pics.sum() > 0:
        recall_pics = (vrais_pics & preds_pics).sum() / vrais_pics.sum() * 100
    if preds_pics.sum() > 0:
        precision_pics = (vrais_pics & preds_pics).sum() / preds_pics.sum() * 100

    print(f"\n\n{'='*60}")
    print(f"  RESULTATS GLOBAUX DU BACKTEST")
    print(f"{'='*60}")
    print(f"  Jours predits     : {len(all_predictions)}")
    print(f"  Re-entrainements  : {n_retrains}")
    print(f"  MAPE globale      : {global_mape:.1f}%")
    print(f"  MAE globale       : {global_mae:,.0f}")
    print(f"  RMSE              : {global_rmse:,.0f}")
    print(f"  Recall pics (>30%): {recall_pics:.1f}%")
    print(f"  Precision pics    : {precision_pics:.1f}%")
    print()

    if global_mape < 15:
        print(f"  VERDICT : EXCELLENT — le modele est fiable")
    elif global_mape < 25:
        print(f"  VERDICT : BON — utilisable en production")
    elif global_mape < 35:
        print(f"  VERDICT : MOYEN — a ameliorer avec plus de donnees")
    else:
        print(f"  VERDICT : FAIBLE — donnees insuffisantes ou trop chaotiques")

    # Sauvegarder
    results = {
        "parametres": {
            "fichier": str(filepath or use_case),
            "train_ratio": train_ratio,
            "step_days": step_days,
            "retrain_every": retrain_every,
        },
        "global": {
            "nb_jours_predits": len(all_predictions),
            "nb_retrainements": n_retrains,
            "mape": round(global_mape, 2),
            "mae": round(global_mae),
            "rmse": round(global_rmse),
            "recall_pics": round(recall_pics, 1),
            "precision_pics": round(precision_pics, 1),
        },
        "par_periode": weekly_results,
        "predictions_vs_reel": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "prevu": round(p),
                "reel": round(a),
                "erreur_pct": round((p - a) / a * 100, 1) if a > 0 else 0,
            }
            for d, p, a in zip(all_dates, all_predictions, all_actuals)
        ],
    }

    output_name = Path(filepath).stem if filepath else use_case
    output_path = RESULTS_DIR / f"backtest_{output_name}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Resultats detailles : {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Backtesting — prediction du passe connu")
    parser.add_argument("--file", help="Fichier de donnees")
    parser.add_argument("--use-case", default="supermarche", help="Use case (si pas de fichier)")
    parser.add_argument("--train-ratio", type=float, default=0.5, help="Ratio train (defaut: 0.5 = 50%%)")
    parser.add_argument("--step", type=int, default=7, help="Pas de prediction en jours (defaut: 7)")
    parser.add_argument("--retrain-every", type=int, default=30, help="Re-entrainer tous les N jours (defaut: 30)")
    args = parser.parse_args()

    backtest(
        filepath=args.file,
        use_case=args.use_case,
        train_ratio=args.train_ratio,
        step_days=args.step,
        retrain_every=args.retrain_every,
    )


if __name__ == "__main__":
    main()
