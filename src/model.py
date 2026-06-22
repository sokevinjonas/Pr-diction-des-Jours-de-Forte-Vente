"""
Entraînement et évaluation des modèles de prédiction.
Modèle principal : XGBoost (rapide, interprétable, léger en production).
Modèle alternatif : Prophet (si peu de features disponibles).
"""

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


def train_xgboost(df, target_col="montant_total", save=True):
    """
    Entraîne un modèle XGBoost avec validation temporelle.
    Retourne le modèle, les métriques et l'importance des features.
    """
    config = load_config()
    test_year = config["model"].get("test_year", 2024)

    feature_cols = get_feature_columns()
    features_disponibles = [c for c in feature_cols if c in df.columns]

    X = df[features_disponibles]
    y = df[target_col]

    # Split temporel : tout sauf la dernière année pour l'entraînement
    mask_train = df["date"].dt.year < test_year
    mask_test = df["date"].dt.year >= test_year

    X_train, X_test = X[mask_train], X[mask_test]
    y_train, y_test = y[mask_train], y[mask_test]

    # Validation croisée temporelle pour le tuning
    tscv = TimeSeriesSplit(n_splits=4)

    model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    # Early stopping avec validation
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Métriques sur le jeu de test
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, df[mask_test], target_col)

    # Validation croisée pour estimer la variance
    cv_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model_cv = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
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
        }, model_path)

    return model, metrics, importance


def compute_metrics(y_true, y_pred, df_test=None, target_col="montant_total"):
    """Calcule les métriques d'évaluation."""
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

    # Accuracy de détection des pics (+30%)
    if df_test is not None and len(df_test) > 30:
        moy_30j = df_test[target_col].rolling(30, min_periods=7).mean()
        seuil_pic = 0.30

        vrais_pics = (y_true.values > moy_30j.values * (1 + seuil_pic))
        preds_pics = (y_pred > moy_30j.values * (1 + seuil_pic))

        # Exclure les NaN du rolling
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
            f"Aucun modèle trouvé. Lancez d'abord l'entraînement."
        )
    return joblib.load(model_path)


def train_pipeline(use_case="supermarche"):
    """Pipeline complet : chargement → features → entraînement → sauvegarde."""
    print(f"[1/3] Chargement des donnees ({use_case})...")
    df = load_ventes(use_case=use_case)

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
