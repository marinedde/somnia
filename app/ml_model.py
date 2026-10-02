"""
Classificateur de stades de sommeil (EEG).

Charge et utilise le pipeline sklearn :
  FeatureExtractor → StandardScaler → RandomForestClassifier

Note : le pipeline sauvegardé contient StandardScaler + RF uniquement.
L'extraction des features est faite ici avant d'appeler le pipeline.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import logging

from app.feature_extractor import FeatureExtractor
from app.model_metrics import charger_metriques

logger = logging.getLogger(__name__)

# Extracteur utilisé pour transformer le signal brut en 16 features
_extractor = FeatureExtractor(fs=100, expected_len=3000)


class SleepStageClassifier:
    """
    Classificateur EEG — 5 stades de sommeil.

    Flux :
        signal brut (3000,) → FeatureExtractor → 16 features → pipeline sklearn
    """

    CLASS_NAMES = {
        0: 'Wake',
        1: 'N1',
        2: 'N2',
        3: 'N3',
        4: 'REM',
    }

    CLASS_INTERPRETATIONS = {
        'Wake': "Éveil — Patient réveillé ou en micro-éveil",
        'N1'  : "Sommeil léger N1 — Endormissement, stade de transition",
        'N2'  : "Sommeil léger N2 — Stade le plus fréquent, fuseaux de sommeil",
        'N3'  : "Sommeil profond N3 — Ondes lentes, récupération physique",
        'REM' : "Sommeil paradoxal REM — Rêves, récupération cognitive",
    }

    MODEL_TYPE = 'Random Forest + Feature Engineering (16 features EEG)'

    @staticmethod
    def _metadata() -> dict:
        """Chiffres lus dans models/training_metrics.json (validation croisée par personne)."""
        m = charger_metriques('eeg')
        metrics = m.get('metrics', {})
        return {
            'model_type'    : SleepStageClassifier.MODEL_TYPE,
            'accuracy'      : metrics.get('accuracy', 0.0),
            'f1_weighted'   : metrics.get('f1_weighted', 0.0),
            'f1_macro'      : metrics.get('f1_macro'),
            'kappa'         : metrics.get('kappa'),
            'n_features'    : 16,
            'training_date' : m.get('training_date', 'inconnue'),
            'dataset'       : m.get('dataset', 'Sleep-EDF Expanded (PhysioNet)'),
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
            raise FileNotFoundError(f"Modèle EEG non trouvé : {self.model_path}")
        logger.info(f"Chargement modèle EEG : {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        # Inférence séquentielle : reproductible au bit près (voir somnia.evaluation.entrainer_rf)
        try:
            self.pipeline.named_steps['clf'].set_params(n_jobs=1)
        except (AttributeError, KeyError, ValueError):
            pass
        logger.info("Modèle EEG chargé")

    def predict(
        self, signal: np.ndarray
    ) -> Tuple[str, int, float, Dict[str, float], str]:
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        if signal.shape[1] != 3000:
            raise ValueError(
                f"Signal EEG doit contenir 3000 points, reçu {signal.shape[1]}"
            )

        # Extraction des 16 features depuis le signal brut
        features = _extractor.transform(signal)  # (1, 16)

        # Prédiction via pipeline (StandardScaler + RF)
        pred_idx      = int(self.pipeline.predict(features)[0])
        pred_class    = self.CLASS_NAMES[pred_idx]
        proba_arr     = self.pipeline.predict_proba(features)[0]
        confidence    = float(np.max(proba_arr))
        probabilities = {
            self.CLASS_NAMES[i]: float(p)
            for i, p in enumerate(proba_arr)
        }
        interpretation = self.CLASS_INTERPRETATIONS[pred_class]

        logger.info(f"EEG → {pred_class} (confiance : {confidence:.2%})")
        return pred_class, pred_idx, confidence, probabilities, interpretation

    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """Retourne le vecteur de features (1, 16) pour monitoring / drift."""
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        return _extractor.transform(signal)

    def get_info(self) -> dict:
        return {
            **self._metadata(),
            'classes'     : list(self.CLASS_NAMES.values()),
            'model_loaded': self.is_loaded(),
        }

    def is_loaded(self) -> bool:
        return self.pipeline is not None
