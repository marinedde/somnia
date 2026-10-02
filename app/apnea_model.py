"""
Détecteur d'apnée du sommeil (ECG).

Note : le pipeline sauvegardé contient StandardScaler + RF uniquement.
L'extraction des features ECG est faite ici avant d'appeler le pipeline.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import logging

from app.ecg_features import ECGFeatureExtractor
from app.model_metrics import charger_metriques

logger = logging.getLogger(__name__)

# Extracteur ECG
_ecg_extractor = ECGFeatureExtractor(fs=100, expected_len=6000)


class ApneaDetector:
    """
    Détecteur ECG — Apnée du sommeil (classification binaire).

    Flux :
        signal brut (6000,) → ECGFeatureExtractor → 16 features → pipeline sklearn
    """

    CLASS_NAMES = {
        0: 'Normal',
        1: 'Apnée',
    }

    MODEL_TYPE = 'Random Forest + Feature Engineering (16 features ECG)'

    @staticmethod
    def _metadata() -> dict:
        """Chiffres lus dans models/training_metrics.json (validation croisée par personne)."""
        m = charger_metriques('ecg')
        metrics = m.get('metrics', {})
        return {
            'model_type'    : ApneaDetector.MODEL_TYPE,
            'auc_roc'       : metrics.get('auc_roc', 0.0),
            'auc_pr'        : metrics.get('auc_pr'),
            'f1_apnea'      : metrics.get('f1_apnee', 0.0),
            'n_features'    : 16,
            'training_date' : m.get('training_date', 'inconnue'),
            'dataset'       : m.get('dataset', 'Apnea-ECG (PhysioNet)'),
            'split_method'  : m.get('split_method', 'inconnu'),
            'metrics_source': m.get('metrics_source', 'inconnue'),
            'ecart_type'    : m.get('ecart_type', {}),
            'avant_fuite'   : m.get('avant_fuite'),
        }

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.pipeline   = None
        self._load()

    def _load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle ECG non trouvé : {self.model_path}")
        logger.info(f"Chargement modèle ECG : {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        # Inférence séquentielle : reproductible au bit près (voir somnia.evaluation.entrainer_rf)
        try:
            self.pipeline.named_steps['clf'].set_params(n_jobs=1)
        except (AttributeError, KeyError, ValueError):
            pass
        logger.info("Modèle ECG chargé")

    def _get_risk(self, confidence: float, predicted_class: str) -> Tuple[str, str]:
        if predicted_class == 'Normal':
            if confidence >= 0.85:
                return 'Faible', "Respiration normale détectée avec haute confiance."
            else:
                return 'Modéré', "Respiration probablement normale — signal ambigu."
        else:
            if confidence >= 0.80:
                return 'Élevé', "Apnée probable — consultation médicale recommandée."
            else:
                return 'Modéré', "Possible apnée détectée — surveillance conseillée."

    def predict(
        self, signal: np.ndarray
    ) -> Tuple[str, int, float, Dict[str, float], str, str]:
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        if signal.shape[1] != 6000:
            raise ValueError(
                f"Signal ECG doit contenir 6000 points, reçu {signal.shape[1]}"
            )

        # Extraction des 16 features depuis le signal brut
        features = _ecg_extractor.transform(signal)  # (1, 16)

        # Prédiction via pipeline (StandardScaler + RF)
        pred_idx      = int(self.pipeline.predict(features)[0])
        pred_class    = self.CLASS_NAMES[pred_idx]
        proba_arr     = self.pipeline.predict_proba(features)[0]
        confidence    = float(np.max(proba_arr))
        probabilities = {
            self.CLASS_NAMES[i]: float(p)
            for i, p in enumerate(proba_arr)
        }
        risk_level, recommendation = self._get_risk(confidence, pred_class)

        logger.info(f"ECG → {pred_class} (confiance : {confidence:.2%} | risque : {risk_level})")
        return pred_class, pred_idx, confidence, probabilities, risk_level, recommendation

    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """Retourne le vecteur de features (1, 16) pour monitoring / drift."""
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        return _ecg_extractor.transform(signal)

    def get_info(self) -> dict:
        return {
            **self._metadata(),
            'classes'     : list(self.CLASS_NAMES.values()),
            'model_loaded': self.is_loaded(),
        }

    def is_loaded(self) -> bool:
        return self.pipeline is not None
