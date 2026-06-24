"""
Modele V2 — Ameliorations :
1. Ensemble XGBoost + LightGBM (moyenne des predictions)
2. Optimisation hyperparametres avec Optuna
3. Classificateur de pics (alerte binaire)
"""

import numpy as np
import pandas as pd
import joblib
import hashlib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    f1_score,
    precision_score,
    recall_score,
)
import xgboost as xgb
import lightgbm as lgb
import optuna

from src.utils import PROJECT_ROOT
from src.feature_engineering import get_feature_columns

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _data_fingerprint(df):
    info = f"{len(df)}_{df['montant_total'].mean():.0f}_{df['montant_total'].std():.0f}_{df['date'].min()}_{df['date'].max()}"
    return hashlib.md5(info.encode()).hexdigest()[:12]


def _get_features(df):
    """Récupère les features disponibles dans le DataFrame."""
    all_cols = get_feature_columns()
    return [c for c in all_cols if c in df.columns]


def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=50):
    """Optimise les hyperparamètres XGBoost avec Optuna."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        return mean_absolute_percentage_error(y_val, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params


def optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=50):
    """Optimise les hyperparamètres LightGBM avec Optuna."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        y_pred = model.predict(X_val)
        return mean_absolute_percentage_error(y_val, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params


def train_ensemble(df, target_col="montant_total", n_trials=50, save=True):
    """
    Entraîne un ensemble XGBoost + LightGBM optimisé avec Optuna.
    Retourne l'ensemble, les métriques, et l'importance des features.
    """
    feature_cols = _get_features(df)
    X = df[feature_cols]
    y = df[target_col]

    # Split temporel 80/20
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Split interne pour Optuna (70% train, 10% val)
    val_idx = int(split_idx * 0.85)
    X_tr, X_val = X_train.iloc[:val_idx], X_train.iloc[val_idx:]
    y_tr, y_val = y_train.iloc[:val_idx], y_train.iloc[val_idx:]

    print(f"  Optimisation XGBoost ({n_trials} essais)...")
    xgb_params = optimize_xgboost(X_tr, y_tr, X_val, y_val, n_trials=n_trials)

    print(f"  Optimisation LightGBM ({n_trials} essais)...")
    lgb_params = optimize_lightgbm(X_tr, y_tr, X_val, y_val, n_trials=n_trials)

    # Entraîner les modèles finaux sur tout le train set
    print("  Entrainement des modeles finaux...")
    xgb_params["random_state"] = 42
    xgb_params["n_jobs"] = -1
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    lgb_params["random_state"] = 42
    lgb_params["n_jobs"] = -1
    lgb_params["verbose"] = -1
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    # Prédictions ensemble (moyenne)
    y_pred_xgb = model_xgb.predict(X_test)
    y_pred_lgb = model_lgb.predict(X_test)
    y_pred_ensemble = (y_pred_xgb + y_pred_lgb) / 2

    # Métriques
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred_ensemble)),
        "mape": float(mean_absolute_percentage_error(y_test, y_pred_ensemble) * 100),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_ensemble))),
        "mape_xgb_seul": float(mean_absolute_percentage_error(y_test, y_pred_xgb) * 100),
        "mape_lgb_seul": float(mean_absolute_percentage_error(y_test, y_pred_lgb) * 100),
    }

    # Importance des features (moyenne des deux modèles)
    imp_xgb = model_xgb.feature_importances_
    imp_lgb = model_lgb.feature_importances_
    imp_combined = (imp_xgb / imp_xgb.sum() + imp_lgb / imp_lgb.sum()) / 2

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": imp_combined,
    }).sort_values("importance", ascending=False)

    if save:
        model_path = MODELS_DIR / "ensemble_model.pkl"
        joblib.dump({
            "model_xgb": model_xgb,
            "model_lgb": model_lgb,
            "features": feature_cols,
            "metrics": metrics,
            "xgb_params": xgb_params,
            "lgb_params": lgb_params,
            "data_fingerprint": _data_fingerprint(df),
        }, model_path)

    return (model_xgb, model_lgb), metrics, importance


def train_pic_classifier(df, target_col="montant_total", seuil_pic=0.30, save=True):
    """
    Entraîne un classificateur binaire : pic (>+30%) ou non.
    Améliore la détection des jours de forte vente.
    """
    feature_cols = _get_features(df)
    X = df[feature_cols]

    # Créer la variable cible binaire
    moy_glissante = df[target_col].rolling(30, min_periods=7).mean()
    y_class = (df[target_col] > moy_glissante * (1 + seuil_pic)).astype(int)

    # Exclure les NaN du rolling
    mask_valid = ~moy_glissante.isna()
    X = X[mask_valid]
    y_class = y_class[mask_valid]

    # Split temporel
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_class.iloc[:split_idx], y_class.iloc[split_idx:]

    # Poids de classe (pics sont rares → sur-pondérer)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)

    y_pred_class = model.predict(X_test)

    metrics_class = {
        "f1_score": float(f1_score(y_test, y_pred_class, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred_class, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_class, zero_division=0)),
        "nb_vrais_pics": int(y_test.sum()),
        "nb_pics_detectes": int(y_pred_class.sum()),
    }

    if save:
        model_path = MODELS_DIR / "pic_classifier.pkl"
        joblib.dump({
            "model": model,
            "features": feature_cols,
            "metrics": metrics_class,
            "seuil_pic": seuil_pic,
        }, model_path)

    return model, metrics_class


def train_v2_pipeline(filepath=None, use_case="supermarche", n_trials=50, with_meteo=True, pays="SN"):
    """Pipeline complet V2 : features améliorées + météo + ensemble + classificateur."""
    from src.data_loader import load_ventes
    from src.feature_engineering import build_features

    print("=" * 60)
    print("  ENTRAINEMENT V2 — Modele ameliore")
    print("=" * 60)

    print(f"\n[1/4] Chargement des donnees...")
    if filepath:
        df = load_ventes(filepath=filepath)
    else:
        df = load_ventes(use_case=use_case)

    print(f"  {len(df)} jours charges")

    print(f"\n[2/4] Feature engineering (v2 — {len(get_feature_columns())} features + meteo={with_meteo})...")
    df = build_features(df, with_meteo=with_meteo, pays=pays)
    print(f"  {len(df)} observations apres transformation")

    print(f"\n[3/4] Entrainement de l'ensemble XGBoost + LightGBM...")
    models, metrics, importance = train_ensemble(df, n_trials=n_trials)

    print(f"\n[4/4] Entrainement du classificateur de pics...")
    _, metrics_class = train_pic_classifier(df)

    print(f"\n{'='*60}")
    print(f"  RESULTATS V2")
    print(f"{'='*60}")
    print(f"  MAPE Ensemble  : {metrics['mape']:.1f}%")
    print(f"  MAPE XGBoost   : {metrics['mape_xgb_seul']:.1f}%")
    print(f"  MAPE LightGBM  : {metrics['mape_lgb_seul']:.1f}%")
    print(f"  MAE            : {metrics['mae']:,.0f}")
    print(f"  RMSE           : {metrics['rmse']:,.0f}")
    print(f"\n  Classificateur de pics :")
    print(f"    Recall       : {metrics_class['recall']*100:.1f}%")
    print(f"    Precision    : {metrics_class['precision']*100:.1f}%")
    print(f"    F1-score     : {metrics_class['f1_score']*100:.1f}%")
    print(f"\n  Top 5 features :")
    for _, row in importance.head(5).iterrows():
        print(f"    - {row['feature']} : {row['importance']:.4f}")

    return models, metrics, metrics_class, importance


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    pays = sys.argv[2] if len(sys.argv) > 2 else "SN"
    train_v2_pipeline(filepath=filepath, n_trials=50, with_meteo=True, pays=pays)
