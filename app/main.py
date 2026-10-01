"""
API FastAPI — Somnia

Deux tâches :
- /predict/sleep-stage : Classification EEG → 5 stades de sommeil
- /predict/apnea       : Détection ECG → Normal / Apnée

Déployé sur HuggingFace Spaces (Docker SDK).
"""

import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    SleepStageRequest, SleepStageResponse,
    ApneaRequest,      ApneaResponse,
    HealthResponse,    ModelInfoResponse,
    ClinicalValidationRequest, ClinicalValidationResponse,
)
from app.ml_model    import SleepStageClassifier
from app.apnea_model import ApneaDetector
from app.monitoring  import SimpleMonitor
from app.validation_store import ValidationStore
from app.baseline import compare_features

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Chemins modèles ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
EEG_MODEL_PATH = PROJECT_ROOT / "models" / "somnia_eeg_pipeline.joblib"
ECG_MODEL_PATH = PROJECT_ROOT / "models" / "somnia_ecg_pipeline.joblib"

# ─── Instances globales ─────────────────────────────────────────────────────
eeg_model : SleepStageClassifier = None
ecg_model : ApneaDetector        = None
monitor   : SimpleMonitor        = SimpleMonitor()
validations: ValidationStore     = ValidationStore()


# ─── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global eeg_model, ecg_model

    logger.info("🚀 Démarrage Somnia...")

    # Chargement modèle EEG
    try:
        eeg_model = SleepStageClassifier(str(EEG_MODEL_PATH))
        logger.info("Modèle EEG chargé")
    except Exception as e:
        logger.error(f"Erreur modèle EEG : {e}")

    # Chargement modèle ECG
    try:
        ecg_model = ApneaDetector(str(ECG_MODEL_PATH))
        logger.info("Modèle ECG chargé")
    except Exception as e:
        logger.error(f"Erreur modèle ECG : {e}")

    yield

    logger.info("Arrêt Somnia v2")


# ─── Application ─────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Somnia API",
    description = """
## Somnia — API de classification du sommeil

Deux modèles de détection automatique basés sur des signaux physiologiques :

### 🧠 Classification EEG — Stades de sommeil
Classifie un signal EEG de 30 secondes en 5 stades :
**Wake**, **N1**, **N2**, **N3** (sommeil profond), **REM**

→ `POST /predict/sleep-stage` — signal de 3000 points (30s × 100Hz)

### ❤️ Détection ECG — Apnée du sommeil
Détecte la présence d'apnée sur un segment ECG de 1 minute :
**Normal** ou **Apnée**

→ `POST /predict/apnea` — signal de 6000 points (60s × 100Hz)

---
**Dataset EEG :** Sleep-EDF Expanded (PhysioNet) — 28 sujets  
**Dataset ECG :** Apnea-ECG (PhysioNet) — 35 sujets  
**Auteur :** Marine Del Dicque | **Version :** 2.0.0
    """,
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─── Root ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "message" : "Bienvenue sur Somnia API",
        "version" : "2.0.0",
        "endpoints": {
            "sleep_stage" : "POST /predict/sleep-stage",
            "apnea"       : "POST /predict/apnea",
            "health"      : "GET  /health",
            "model_info"  : "GET  /model-info",
            "monitoring"  : "GET  /monitoring/stats",
            "validations" : "POST /validations",
            "docs"        : "GET  /docs",
        }
    }


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Vérifie l'état de santé de l'API et des deux modèles."""
    return HealthResponse(
        status           = "healthy" if (
            eeg_model and eeg_model.is_loaded() and
            ecg_model and ecg_model.is_loaded()
        ) else "degraded",
        eeg_model_loaded = bool(eeg_model and eeg_model.is_loaded()),
        ecg_model_loaded = bool(ecg_model and ecg_model.is_loaded()),
        version          = "2.0.0",
    )


# ─── Model info ───────────────────────────────────────────────────────────────
@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
async def get_model_info():
    """Retourne les métadonnées des deux modèles."""
    if not (eeg_model and eeg_model.is_loaded()):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Modèle EEG non chargé")
    if not (ecg_model and ecg_model.is_loaded()):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Modèle ECG non chargé")

    eeg_info = eeg_model.get_info()
    ecg_info = ecg_model.get_info()

    return ModelInfoResponse(
        eeg_model_type  = eeg_info['model_type'],
        eeg_accuracy    = eeg_info['accuracy'],
        eeg_f1_weighted = eeg_info['f1_weighted'],
        eeg_n_features  = eeg_info['n_features'],
        eeg_classes     = eeg_info['classes'],
        ecg_model_type  = ecg_info['model_type'],
        ecg_auc_roc     = ecg_info['auc_roc'],
        ecg_f1_apnea    = ecg_info['f1_apnea'],
        ecg_n_features  = ecg_info['n_features'],
        ecg_classes     = ecg_info['classes'],
        training_date   = eeg_info['training_date'],
        dataset_eeg     = eeg_info['dataset'],
        dataset_ecg     = ecg_info['dataset'],
    )


# ─── Predict sleep stage ──────────────────────────────────────────────────────
@app.post(
    "/predict/sleep-stage",
    response_model = SleepStageResponse,
    tags           = ["Prediction"],
    summary        = "Classifie un signal EEG en stade de sommeil",
)
async def predict_sleep_stage(request: SleepStageRequest):
    """
    Classifie un signal EEG de 30 secondes (3000 points à 100Hz).

    **Retourne :**
    - `predicted_class`  : Wake | N1 | N2 | N3 | REM
    - `confidence`       : Probabilité de la classe prédite (0-1)
    - `probabilities`    : Probabilités pour chaque stade
    - `interpretation`   : Description clinique du stade
    """
    if not (eeg_model and eeg_model.is_loaded()):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Modèle EEG non disponible")
    start = time.time()
    try:
        signal = np.array(request.signal, dtype=np.float32).reshape(1, -1)
        pred_class, pred_idx, confidence, probabilities, interpretation = \
            eeg_model.predict(signal)
        features = eeg_model.extract_features(signal)

        processing_ms = (time.time() - start) * 1000
        monitor.log_prediction(
            task              = 'sleep_stage',
            prediction        = pred_class,
            confidence        = confidence,
            probabilities     = probabilities,
            processing_time_ms= processing_ms,
            features          = features,
        )

        return SleepStageResponse(
            predicted_class = pred_class,
            predicted_index = pred_idx,
            confidence      = confidence,
            probabilities   = probabilities,
            interpretation  = interpretation,
        )
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            detail=f"Signal invalide : {e}")
    except Exception as e:
        logger.error(f"Erreur prédiction EEG : {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Erreur interne : {e}")


# ─── Predict apnea ────────────────────────────────────────────────────────────
@app.post(
    "/predict/apnea",
    response_model = ApneaResponse,
    tags           = ["Prediction"],
    summary        = "Détecte la présence d'apnée sur un signal ECG",
)
async def predict_apnea(request: ApneaRequest):
    """
    Détecte la présence d'apnée sur un signal ECG de 60 secondes (6000 points à 100Hz).

    **Retourne :**
    - `predicted_class`  : Normal | Apnée
    - `confidence`       : Confiance de la prédiction (0-1)
    - `probabilities`    : Probabilités Normal/Apnée
    - `risk_level`       : Faible | Modéré | Élevé
    - `recommendation`   : Recommandation clinique
    """
    if not (ecg_model and ecg_model.is_loaded()):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Modèle ECG non disponible")
    start = time.time()
    try:
        signal = np.array(request.signal, dtype=np.float32).reshape(1, -1)
        pred_class, pred_idx, confidence, probabilities, risk_level, recommendation = \
            ecg_model.predict(signal)
        features = ecg_model.extract_features(signal)

        processing_ms = (time.time() - start) * 1000
        monitor.log_prediction(
            task              = 'apnea',
            prediction        = pred_class,
            confidence        = confidence,
            probabilities     = probabilities,
            processing_time_ms= processing_ms,
            features          = features,
        )

        return ApneaResponse(
            predicted_class = pred_class,
            predicted_index = pred_idx,
            confidence      = confidence,
            probabilities   = probabilities,
            risk_level      = risk_level,
            recommendation  = recommendation,
        )
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            detail=f"Signal invalide : {e}")
    except Exception as e:
        logger.error(f"Erreur prédiction ECG : {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Erreur interne : {e}")


# ─── Monitoring ───────────────────────────────────────────────────────────────
@app.get("/monitoring/stats", tags=["Monitoring"])
async def get_stats(last_n: int = 100):
    """Statistiques des N dernières prédictions."""
    return monitor.get_statistics(last_n)


@app.get("/monitoring/drift", tags=["Monitoring"])
async def check_drift(threshold: float = 0.1, window_size: int = 50):
    """Détection de drift de confiance sur les prédictions récentes."""
    return monitor.detect_drift(threshold, window_size)


@app.get("/monitoring/drift/features", tags=["Monitoring"])
async def check_feature_drift(
    task: str = "sleep_stage",
    threshold: float = 2.0,
    window_size: int = 30,
):
    """
    Drift des features vs baseline d'entraînement (models/baseline_stats.json).
    task : sleep_stage | apnea
    """
    import numpy as np
    task_key = "sleep_stage" if task == "sleep_stage" else "apnea"
    feats = monitor.get_recent_features(task_key, window_size)
    if feats is None or len(feats) < 5:
        return {
            "drift_detected": False,
            "message": f"Pas assez de prédictions avec features ({0 if feats is None else len(feats)})",
            "threshold": threshold,
        }
    return compare_features(np.array(feats), task_key, threshold)


@app.post(
    "/validations",
    response_model=ClinicalValidationResponse,
    tags=["Clinical"],
    summary="Enregistre une validation médecin",
)
async def save_clinical_validation(body: ClinicalValidationRequest):
    """Audit trail des validations humaines (JSONL persistant)."""
    entry = validations.append(
        task=body.task,
        model_prediction=body.model_prediction,
        model_confidence=body.model_confidence,
        clinician_verdict=body.clinician_verdict,
        comment=body.comment,
        extra=body.extra,
    )
    return ClinicalValidationResponse(**entry)


@app.get("/validations/recent", tags=["Clinical"])
async def list_clinical_validations(n: int = 20, task: str | None = None):
    """Dernières validations enregistrées."""
    return {
        "count_total": validations.count(),
        "validations": validations.list_recent(n, task),
    }


@app.get("/monitoring/recent", tags=["Monitoring"])
async def get_recent(n: int = 10):
    """N dernières prédictions."""
    return {"predictions": monitor.get_recent_logs(n)}


@app.delete("/monitoring/reset", tags=["Monitoring"])
async def reset_monitoring():
    """Réinitialise les logs de monitoring."""
    monitor.reset()
    return {"message": "Monitoring réinitialisé"}


# ─── Dev ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
