"""
Schemas Pydantic pour l'API Somnia.

Définit les modèles de requête et réponse pour :
- /predict/sleep-stage (EEG → 5 stades)
- /predict/apnea       (ECG → Normal / Apnée)
- /health
- /model-info
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional, List


# ============================================================================
# REQUÊTES
# ============================================================================

class SleepStageRequest(BaseModel):
    """Requête pour la classification des stades de sommeil."""
    signal: List[float] = Field(
        ...,
        description="Signal EEG de 30 secondes — 3000 valeurs à 100 Hz",
        min_length=3000,
        max_length=3000,
    )

    @field_validator('signal')
    @classmethod
    def check_signal_length(cls, v):
        if len(v) != 3000:
            raise ValueError(
                f"Le signal EEG doit contenir exactement 3000 points "
                f"(30s × 100Hz), reçu {len(v)}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "signal": [0.0] * 3000
            }
        }
    }


class ApneaRequest(BaseModel):
    """Requête pour la détection d'apnée."""
    signal: List[float] = Field(
        ...,
        description="Signal ECG de 60 secondes — 6000 valeurs à 100 Hz",
        min_length=6000,
        max_length=6000,
    )

    @field_validator('signal')
    @classmethod
    def check_signal_length(cls, v):
        if len(v) != 6000:
            raise ValueError(
                f"Le signal ECG doit contenir exactement 6000 points "
                f"(60s × 100Hz), reçu {len(v)}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "signal": [0.0] * 6000
            }
        }
    }


# ============================================================================
# RÉPONSES
# ============================================================================

class SleepStageResponse(BaseModel):
    """Réponse de classification des stades de sommeil."""
    predicted_class : str   = Field(..., description="Stade prédit : Wake, N1, N2, N3, REM")
    predicted_index : int   = Field(..., description="Index de classe : 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM")
    confidence      : float = Field(..., description="Confiance de la prédiction (0-1)")
    probabilities   : Dict[str, float] = Field(
        ..., description="Probabilités pour chaque stade"
    )
    interpretation  : str   = Field(..., description="Interprétation clinique du stade")


class ApneaResponse(BaseModel):
    """Réponse de détection d'apnée."""
    predicted_class : str   = Field(..., description="Normal ou Apnée")
    predicted_index : int   = Field(..., description="0=Normal, 1=Apnée")
    confidence      : float = Field(..., description="Confiance de la prédiction (0-1)")
    probabilities   : Dict[str, float] = Field(
        ..., description="Probabilités Normal/Apnée"
    )
    risk_level      : str   = Field(..., description="Niveau de risque : Faible, Modéré, Élevé")
    recommendation  : str   = Field(..., description="Recommandation clinique")


class HealthResponse(BaseModel):
    """Réponse de l'endpoint de santé."""
    status        : str  = Field(..., description="healthy ou unhealthy")
    eeg_model_loaded : bool = Field(..., description="Modèle EEG chargé")
    ecg_model_loaded : bool = Field(..., description="Modèle ECG chargé")
    version       : str  = Field(default="2.0.0")


class ModelInfoResponse(BaseModel):
    """Informations sur les modèles chargés."""
    # EEG
    eeg_model_type    : str
    eeg_accuracy      : float
    eeg_f1_weighted   : float
    eeg_n_features    : int
    eeg_classes       : List[str]
    # ECG
    ecg_model_type    : str
    ecg_auc_roc       : float
    ecg_f1_apnea      : float
    ecg_n_features    : int
    ecg_classes       : List[str]
    # Général
    training_date     : str
    dataset_eeg       : str
    dataset_ecg       : str


class MonitoringStatsResponse(BaseModel):
    """Statistiques de monitoring."""
    total_predictions    : int
    eeg_predictions      : int
    ecg_predictions      : int
    avg_confidence       : Optional[float]
    avg_processing_time_ms: Optional[float]
    class_distribution   : Dict[str, int]


class ClinicalValidationRequest(BaseModel):
    """Validation médecin sur une prédiction Somnia."""
    task: str = Field(..., description="sleep_stage ou apnea")
    model_prediction: str
    model_confidence: float = Field(..., ge=0.0, le=1.0)
    clinician_verdict: str = Field(
        ..., description="Confirmé, Incorrect ou Ambigu"
    )
    comment: str = ""
    extra: Optional[Dict] = None

    @field_validator('task')
    @classmethod
    def check_task(cls, v):
        if v not in ('sleep_stage', 'apnea'):
            raise ValueError("task doit être 'sleep_stage' ou 'apnea'")
        return v


class ClinicalValidationResponse(BaseModel):
    id: str
    timestamp: str
    task: str
    model_prediction: str
    model_confidence: float
    clinician_verdict: str
    comment: str
