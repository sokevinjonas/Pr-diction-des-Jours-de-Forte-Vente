"""
Observabilité du système ML — Logging et tracking des prédictions.
Enregistre chaque prédiction et permet de comparer avec la réalité.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "data" / "processed" / "predictions.db"


def init_db():
    """Crée les tables SQLite pour le tracking."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table des prédictions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            date_prediction TEXT NOT NULL,
            use_case TEXT NOT NULL,
            montant_predit REAL,
            montant_reel REAL,
            erreur_pct REAL,
            alerte_pic INTEGER,
            features_json TEXT,
            model_version TEXT
        )
    """)

    # Table des métriques globales (calculée quotidiennement)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_calcul TEXT NOT NULL,
            use_case TEXT NOT NULL,
            mape REAL,
            mae REAL,
            rmse REAL,
            nb_predictions INTEGER,
            model_version TEXT
        )
    """)

    # Table des alertes (quand le modèle se dégrade)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            use_case TEXT NOT NULL,
            type_alerte TEXT,
            message TEXT,
            metric_name TEXT,
            metric_value REAL,
            seuil REAL
        )
    """)

    conn.commit()
    conn.close()


def log_prediction(date_prediction, use_case, montant_predit, montant_reel=None,
                   alerte_pic=False, features=None, model_version="v2"):
    """
    Enregistre une prédiction dans la base de données.

    Args:
        date_prediction : date pour laquelle on prédit
        use_case : nom du commerce (supermarche, restaurant, etc)
        montant_predit : montant prédit par le modèle
        montant_reel : montant réel (None si pas encore connu)
        alerte_pic : booléen, True si c'est un pic détecté
        features : dict des features utilisées
        model_version : v1, v2, etc
    """
    init_db()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    erreur_pct = None
    if montant_reel is not None and montant_predit > 0:
        erreur_pct = abs(montant_predit - montant_reel) / montant_reel * 100

    features_json = json.dumps(features) if features else None

    cursor.execute("""
        INSERT INTO predictions
        (timestamp, date_prediction, use_case, montant_predit, montant_reel,
         erreur_pct, alerte_pic, features_json, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        date_prediction,
        use_case,
        float(montant_predit),
        float(montant_reel) if montant_reel else None,
        erreur_pct,
        int(alerte_pic),
        features_json,
        model_version,
    ))

    conn.commit()
    conn.close()


def get_predictions(use_case=None, days=30, include_nulls=False):
    """Récupère les prédictions enregistrées."""
    init_db()

    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM predictions WHERE timestamp > datetime('now', '-' || ? || ' days')"
    params = [days]

    if use_case:
        query += " AND use_case = ?"
        params.append(use_case)

    if not include_nulls:
        query += " AND montant_reel IS NOT NULL"

    query += " ORDER BY timestamp DESC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    return df


def compute_daily_metrics(use_case, date_str=None, model_version="v2"):
    """
    Calcule et enregistre les métriques quotidiennes.
    Compare les prédictions avec la réalité.
    """
    init_db()

    conn = sqlite3.connect(DB_PATH)

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Récupérer les prédictions du jour avec réalité
    query = """
        SELECT montant_predit, montant_reel, erreur_pct
        FROM predictions
        WHERE use_case = ?
          AND date_prediction = ?
          AND montant_reel IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=[use_case, date_str])

    if df.empty:
        conn.close()
        return None

    # Calculer les métriques
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

    mape = mean_absolute_percentage_error(df["montant_reel"], df["montant_predit"]) * 100
    mae = mean_absolute_error(df["montant_reel"], df["montant_predit"])
    rmse = np.sqrt(mean_squared_error(df["montant_reel"], df["montant_predit"]))

    # Enregistrer
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO metrics
        (date_calcul, use_case, mape, mae, rmse, nb_predictions, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        date_str, use_case, mape, mae, rmse, len(df), model_version
    ))

    conn.commit()
    conn.close()

    return {
        "date": date_str,
        "use_case": use_case,
        "mape": mape,
        "mae": mae,
        "rmse": rmse,
        "nb_predictions": len(df),
    }


def get_metrics_history(use_case, days=90):
    """Récupère l'historique des métriques."""
    init_db()

    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT date_calcul, mape, mae, rmse, nb_predictions
        FROM metrics
        WHERE use_case = ? AND date_calcul > datetime('now', '-' || ? || ' days')
        ORDER BY date_calcul ASC
    """
    df = pd.read_sql_query(query, conn, params=[use_case, days])
    conn.close()

    return df


def log_alert(use_case, alert_type, message, metric_name=None, metric_value=None, seuil=None):
    """Enregistre une alerte (dérive de performance, etc)."""
    init_db()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO alerts
        (timestamp, use_case, type_alerte, message, metric_name, metric_value, seuil)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        use_case,
        alert_type,
        message,
        metric_name,
        metric_value,
        seuil,
    ))

    conn.commit()
    conn.close()


def get_alerts(use_case=None, days=7, alert_type=None):
    """Récupère les alertes récentes."""
    init_db()

    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM alerts WHERE timestamp > datetime('now', '-' || ? || ' days')"
    params = [days]

    if use_case:
        query += " AND use_case = ?"
        params.append(use_case)

    if alert_type:
        query += " AND type_alerte = ?"
        params.append(alert_type)

    query += " ORDER BY timestamp DESC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    return df


def export_metrics(use_case, output_path=None):
    """Exporte les métriques en CSV."""
    if output_path is None:
        output_path = PROJECT_ROOT / f"metriques_{use_case}.csv"

    df = get_metrics_history(use_case, days=365)
    df.to_csv(output_path, index=False)

    return output_path
