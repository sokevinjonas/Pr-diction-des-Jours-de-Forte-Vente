"""
Entraînement et évaluation des modèles de prédiction.
Modèle auto-adaptatif : se réentraîne automatiquement quand les données changent.
"""

import hashlib
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
import xgboost as xgb

from src.utils import PROJECT_ROOT, load_config
from src.data_loader import load_ventes
from src.feature_engineering import build_features, get_feature_columns

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _data_fingerprint(df):
    """Crée une empreinte des données pour détecter les changements."""
    info = f"{len(df)}_{df['montant_total'].mean():.0f}_{df['montant_total'].std():.0f}_{df['date'].min()}_{df['date'].max()}"
    return hashlib.md5(info.encode()).hexdigest()[:12]


def _model_is_compatible(df):
    """Vérifie si le modèle sauvegardé est compatible avec les données actuelles."""
    model_path = MODELS_DIR / "xgboost_model.pkl"
    if not model_path.exists():
        return False

    model_data = joblib.load(model_path)
    saved_fingerprint = model_data.get("data_fingerprint")
    current_fingerprint = _data_fingerprint(df)

    if saved_fingerprint != current_fingerprint:
        return False

    return True


def train_xgboost(df, target_col="montant_total", save=True):
    """
    Entraîne un modèle XGBoost avec validation temporelle.
    S'adapte automatiquement à la taille et la structure des données.
    """
    feature_cols = get_feature_columns()
    features_disponibles = [c for c in feature_cols if c in df.columns]

    X = df[features_disponibles]
    y = df[target_col]

    # Split temporel adaptatif : les 20% les plus récents pour le test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Adapter les hyperparamètres à la taille des données
    n_samples = len(X_train)
    if n_samples < 100:
        n_estimators = 100
        max_depth = 3
        learning_rate = 0.1
    elif n_samples < 500:
        n_estimators = 300
        max_depth = 5
        learning_rate = 0.06
    else:
        n_estimators = 600
        max_depth = 6
        learning_rate = 0.04

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Métriques sur le jeu de test
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, df.iloc[split_idx:], target_col)

    # Validation croisée
    n_splits = min(4, max(2, n_samples // 100))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model_cv = xgb.XGBRegressor(
            n_estimators=min(300, n_estimators),
            learning_rate=0.05, max_depth=max_depth,
            random_state=42, n_jobs=-1,
        )
        model_cv.fit(X_cv_train, y_cv_train, verbose=False)
        y_cv_pred = model_cv.predict(X_cv_val)
        cv_scores.append(mean_absolute_percentage_error(y_cv_val, y_cv_pred))

    metrics["cv_mape_mean"] = np.mean(cv_scores)
    metrics["cv_mape_std"] = np.std(cv_scores)

    # Importance des features
    importance = pd.DataFrame({
        "feature": features_disponibles,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    if save:
        model_path = MODELS_DIR / "xgboost_model.pkl"
        joblib.dump({
            "model": model,
            "features": features_disponibles,
            "metrics": metrics,
            "data_fingerprint": _data_fingerprint(df),
            "data_stats": {
                "mean": float(df[target_col].mean()),
                "std": float(df[target_col].std()),
                "n_rows": len(df),
                "date_min": str(df["date"].min().date()),
                "date_max": str(df["date"].max().date()),
            },
        }, model_path)

    return model, metrics, importance


def auto_train_if_needed(df):
    """
    Vérifie si le modèle existant est compatible avec les données.
    Si non, réentraîne automatiquement. Retourne le modèle prêt.
    """
    if _model_is_compatible(df):
        return joblib.load(MODELS_DIR / "xgboost_model.pkl")

    # Réentraînement automatique
    df_features = build_features(df)

    if len(df_features) < 30:
        raise ValueError(
            f"Pas assez de donnees pour entrainer un modele ({len(df_features)} jours). "
            f"Minimum requis : 30 jours."
        )

    train_xgboost(df_features, save=True)
    return joblib.load(MODELS_DIR / "xgboost_model.pkl")


def compute_metrics(y_true, y_pred, df_test=None, target_col="montant_total"):
    """Calcule les métriques d'évaluation."""
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

    if df_test is not None and len(df_test) > 30:
        moy_30j = df_test[target_col].rolling(30, min_periods=7).mean()
        seuil_pic = 0.30

        vrais_pics = (y_true.values > moy_30j.values * (1 + seuil_pic))
        preds_pics = (y_pred > moy_30j.values * (1 + seuil_pic))

        mask_valid = ~np.isnan(moy_30j.values)
        if mask_valid.sum() > 0:
            vrais_pics = vrais_pics[mask_valid]
            preds_pics = preds_pics[mask_valid]

            if vrais_pics.sum() > 0:
                recall_pics = (vrais_pics & preds_pics).sum() / vrais_pics.sum()
                metrics["recall_pics"] = recall_pics * 100
            else:
                metrics["recall_pics"] = None

            if preds_pics.sum() > 0:
                precision_pics = (vrais_pics & preds_pics).sum() / preds_pics.sum()
                metrics["precision_pics"] = precision_pics * 100
            else:
                metrics["precision_pics"] = None

    return metrics


def load_trained_model(use_case="supermarche"):
    """Charge un modèle entraîné depuis le disque."""
    model_path = MODELS_DIR / "xgboost_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "Aucun modele trouve. Lancez d'abord l'entrainement."
        )
    return joblib.load(model_path)


def train_pipeline(use_case="supermarche", filepath=None):
    """Pipeline complet : chargement → features → entraînement → sauvegarde."""
    print(f"[1/3] Chargement des donnees ({use_case})...")
    df = load_ventes(use_case=use_case, filepath=filepath)

    print("[2/3] Feature engineering...")
    df = build_features(df)

    print(f"[3/3] Entrainement XGBoost ({len(df)} observations)...")
    model, metrics, importance = train_xgboost(df)

    print("\n=== Resultats ===")
    print(f"  MAE  : {metrics['mae']:,.0f} FCFA")
    print(f"  MAPE : {metrics['mape']:.1f}%")
    print(f"  RMSE : {metrics['rmse']:,.0f} FCFA")
    if metrics.get("recall_pics"):
        print(f"  Recall pics : {metrics['recall_pics']:.1f}%")
    print(f"  CV MAPE : {metrics['cv_mape_mean']:.1f}% (+/- {metrics['cv_mape_std']:.1f}%)")
    print(f"\n  Top 5 features :")
    for _, row in importance.head(5).iterrows():
        print(f"    - {row['feature']} : {row['importance']:.3f}")

    return model, metrics, importance


if __name__ == "__main__":
    train_pipeline(use_case="supermarche")
