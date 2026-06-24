"""
API REST — FastAPI pour prédictions ML en production.
Endpoints pour prédictions, monitoring, et admin.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timedelta
import time
import logging

from src.data_loader import load_ventes
from src.feature_engineering import build_features
from src.model_v2 import train_v2_pipeline
from src.observability import log_prediction, compute_daily_metrics, get_metrics_history
from src.monitoring import ModelMonitor
from src.security import InputValidator, OutputFilter
from src.audit import log_prediction_audit, log_api_access, export_compliance_report
from src.rate_limit import RATE_LIMITER_PREDICT, RATE_LIMITER_API

# === Configuration ===
app = FastAPI(
    title="Prédiction Ventes ML API",
    description="API de prédiction pour jours de forte vente",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Models ===

class Features(BaseModel):
    jour_semaine: int = Field(..., ge=0, le=6, description="0=Lundi, 6=Dimanche")
    jour_mois: int = Field(..., ge=1, le=31, description="Jour du mois")
    mois: int = Field(..., ge=1, le=12, description="Mois de l'année")
    temperature_max: Optional[float] = None
    precipitation_mm: Optional[float] = None
    ventes_j_1: Optional[float] = None


class PredictionRequest(BaseModel):
    use_case: str = Field(..., description="supermarche, restaurant, mobile_money, grossiste")
    date_prediction: str = Field(..., description="YYYY-MM-DD")
    features: Features


class PredictionResponse(BaseModel):
    prediction: float
    unit: str = "FCFA"
    confidence: float
    alerte_pic: bool
    timestamp: str


class HealthCheckResponse(BaseModel):
    health: str
    mape: float
    mae: float
    coverage_pct: float


class MetricsResponse(BaseModel):
    date: str
    mape: float
    mae: float
    rmse: float


# === Middleware ===

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Log tous les accès API."""
    start_time = time.time()

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000

    user_id = request.headers.get("X-User-ID", "anonymous")
    client_ip = request.client.host if request.client else "unknown"

    log_api_access(
        endpoint=str(request.url.path),
        method=request.method,
        status_code=response.status_code,
        response_time_ms=process_time,
        user_id=user_id,
        ip_address=client_ip,
    )

    return response


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Health check API."""
    return {
        "service": "Prédiction Ventes ML",
        "version": "2.0.0",
        "status": "running",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, pred_request: PredictionRequest):
    """
    Effectuer une prédiction pour un jour donné.

    - **use_case** : Type de commerce
    - **date_prediction** : Date au format YYYY-MM-DD
    - **features** : Features pour la prédiction
    """
    user_id = request.headers.get("X-User-ID", "anonymous")
    client_ip = request.client.host if request.client else "unknown"

    # Rate limiting
    allowed, reason = RATE_LIMITER_PREDICT.check_rate_limit(user_id, "predict")

    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    # Validation entrées
    validator = InputValidator(pred_request.use_case)

    features_dict = pred_request.features.dict()

    is_valid, error = validator.validate_features(features_dict)

    if not is_valid:
        log_prediction_audit(
            use_case=pred_request.use_case,
            prediction=0,
            validation_status="rejected_invalid_input",
            security_check_passed=False,
            user_id=user_id,
        )
        raise HTTPException(status_code=400, detail=f"Invalid features: {error}")

    # Détection d'anomalies
    anomaly, reason = validator.detect_anomaly(features_dict)

    if anomaly:
        logger.warning(f"Anomaly detected for {user_id}: {reason}")

    # Charger le modèle (en vrai, on le chargerait depuis disk)
    try:
        df = load_ventes(use_case=pred_request.use_case)
        df = build_features(df, with_meteo=False)  # Faster pour API

        models, metrics, _, _ = train_v2_pipeline(
            use_case=pred_request.use_case, n_trials=10, with_meteo=False
        )

        model_xgb, model_lgb = models

        # Simuler une prédiction
        import numpy as np

        X = np.array([[
            features_dict.get("jour_semaine", 0),
            features_dict.get("jour_mois", 15),
            features_dict.get("mois", 6),
        ]])

        # Ensemble prediction
        pred_xgb = model_xgb.predict(X)[0]
        pred_lgb = model_lgb.predict(X)[0]

        prediction = (pred_xgb + pred_lgb) / 2

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prédiction échouée")

    # Filtering de sortie
    output_filter = OutputFilter(pred_request.use_case)

    prediction, is_valid, warning = output_filter.filter_prediction(prediction)

    if not is_valid:
        raise HTTPException(status_code=500, detail="Prédiction invalide")

    # Détection de pic (confiance > 60%)
    alerte_pic = prediction > (df["montant_total"].mean() * 1.3)

    # Logging
    log_prediction(
        date_prediction=pred_request.date_prediction,
        use_case=pred_request.use_case,
        montant_predit=prediction,
        alerte_pic=alerte_pic,
        features=features_dict,
        model_version="v2",
    )

    log_prediction_audit(
        use_case=pred_request.use_case,
        prediction=prediction,
        validation_status="validated",
        security_check_passed=True,
        anomaly_detected=anomaly,
        model_version="v2",
        user_id=user_id,
    )

    return PredictionResponse(
        prediction=float(prediction),
        confidence=0.85,
        alerte_pic=bool(alerte_pic),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/health/{use_case}", response_model=HealthCheckResponse)
async def health_check(use_case: str):
    """
    Vérifier la santé du modèle.
    """
    try:
        monitor = ModelMonitor(use_case, baseline_mape=20.0)
        health = monitor.run_full_check()

        perf = health["checks"]["performance_drift"]
        coverage = health["checks"]["prediction_coverage"]

        metrics_hist = get_metrics_history(use_case, days=7)

        mape = metrics_hist["mape"].mean() if not metrics_hist.empty else 0
        mae = metrics_hist["mae"].mean() if not metrics_hist.empty else 0

        return HealthCheckResponse(
            health=health["health"],
            mape=mape,
            mae=mae,
            coverage_pct=coverage.get("coverage_pct", 0),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/{use_case}", response_model=List[MetricsResponse])
async def get_metrics(use_case: str, days: int = 30):
    """
    Récupérer l'historique des métriques.
    """
    try:
        metrics_df = get_metrics_history(use_case, days=days)

        return [
            MetricsResponse(
                date=row["date_calcul"],
                mape=float(row["mape"]),
                mae=float(row["mae"]),
                rmse=float(row["rmse"]),
            )
            for _, row in metrics_df.iterrows()
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain/{use_case}")
async def retrain_model(use_case: str, request: Request):
    """
    Réentraîner le modèle (admin seulement).
    """
    # Admin check (simple exemple)
    auth_token = request.headers.get("X-API-Key", "")

    if auth_token != "admin_secret_key":
        raise HTTPException(status_code=403, detail="Unauthorized")

    try:
        models, metrics, metrics_class, _ = train_v2_pipeline(
            use_case=use_case, n_trials=50, with_meteo=True, pays="SN"
        )

        return {
            "status": "success",
            "mape": metrics["mape"],
            "mae": metrics["mae"],
            "recall_pics": metrics_class["recall"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compliance")
async def compliance_report(request: Request):
    """
    Exporter un rapport de conformité (RGPD/CCPA).
    """
    auth_token = request.headers.get("X-API-Key", "")

    if auth_token != "admin_secret_key":
        raise HTTPException(status_code=403, detail="Unauthorized")

    try:
        report_path = export_compliance_report()

        return {
            "status": "success",
            "report_path": str(report_path),
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Error Handlers ===

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom error handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
