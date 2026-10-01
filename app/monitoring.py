"""
Monitoring simple pour Somnia.

Enregistre les prédictions en mémoire et fournit des statistiques.
Pas de base de données — adapté pour un déploiement HuggingFace Spaces.
"""

from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np


class SimpleMonitor:
    """
    Moniteur de prédictions en mémoire.

    Stocke les N dernières prédictions et calcule des statistiques
    de distribution, confiance et temps de traitement.
    """

    MAX_LOGS = 1000  # Limite mémoire

    def __init__(self):
        self._logs: List[dict] = []

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        task          : str,    # 'sleep_stage' ou 'apnea'
        prediction    : str,
        confidence    : float,
        probabilities : Dict[str, float],
        processing_time_ms: float,
        features      : Optional[np.ndarray] = None,
    ) -> None:
        """Enregistre une prédiction (features optionnelles pour drift)."""
        entry = {
            'timestamp' : datetime.now(timezone.utc).isoformat(),
            'task'             : task,
            'prediction'       : prediction,
            'confidence'       : round(confidence, 4),
            'probabilities'    : probabilities,
            'processing_time_ms': round(processing_time_ms, 2),
        }
        if features is not None:
            entry['features'] = np.asarray(features, dtype=np.float32).reshape(-1).tolist()
        self._logs.append(entry)
        # Garder seulement les MAX_LOGS derniers
        if len(self._logs) > self.MAX_LOGS:
            self._logs = self._logs[-self.MAX_LOGS:]

    # ------------------------------------------------------------------
    # Statistiques
    # ------------------------------------------------------------------

    def get_statistics(self, last_n: int = 100) -> dict:
        """Calcule les statistiques sur les N dernières prédictions."""
        logs = self._logs[-last_n:]

        if not logs:
            return {
                'total_predictions'    : 0,
                'eeg_predictions'      : 0,
                'ecg_predictions'      : 0,
                'avg_confidence'       : None,
                'avg_processing_time_ms': None,
                'class_distribution'   : {},
                'message'              : 'Aucune prédiction enregistrée',
            }

        eeg_logs = [l for l in logs if l['task'] == 'sleep_stage']
        ecg_logs = [l for l in logs if l['task'] == 'apnea']

        class_dist: Dict[str, int] = defaultdict(int)
        for log in logs:
            class_dist[f"{log['task']}:{log['prediction']}"] += 1

        return {
            'total_predictions'    : len(logs),
            'eeg_predictions'      : len(eeg_logs),
            'ecg_predictions'      : len(ecg_logs),
            'avg_confidence'       : round(
                float(np.mean([l['confidence'] for l in logs])), 4
            ),
            'avg_processing_time_ms': round(
                float(np.mean([l['processing_time_ms'] for l in logs])), 2
            ),
            'class_distribution'   : dict(class_dist),
            'confidence_stats'     : {
                'min' : round(float(min(l['confidence'] for l in logs)), 4),
                'max' : round(float(max(l['confidence'] for l in logs)), 4),
                'std' : round(float(np.std([l['confidence'] for l in logs])), 4),
            },
        }

    def detect_drift(
        self,
        threshold  : float = 0.1,
        window_size: int   = 50,
    ) -> dict:
        """
        Détection simple de drift.

        Compare la confiance moyenne de la première moitié
        vs la seconde moitié des N dernières prédictions.
        """
        logs = self._logs[-window_size * 2:]

        if len(logs) < window_size * 2:
            return {
                'drift_detected': False,
                'message'       : f'Données insuffisantes ({len(logs)}/{window_size*2})',
                'threshold'     : threshold,
            }

        first_half  = logs[:window_size]
        second_half = logs[window_size:]

        conf_first  = float(np.mean([l['confidence'] for l in first_half]))
        conf_second = float(np.mean([l['confidence'] for l in second_half]))
        delta       = abs(conf_first - conf_second)
        drift       = delta > threshold

        return {
            'drift_detected'   : drift,
            'confidence_first' : round(conf_first,  4),
            'confidence_second': round(conf_second, 4),
            'delta'            : round(delta, 4),
            'threshold'        : threshold,
            'message'          : (
                f"Drift détecté (delta={delta:.3f} > {threshold})"
                if drift else
                f"Pas de drift (delta={delta:.3f} < {threshold})"
            ),
        }

    def get_recent_logs(self, n: int = 10) -> List[dict]:
        """Retourne les N dernières prédictions."""
        return self._logs[-n:]

    def get_recent_features(self, task: str, n: int = 50) -> Optional[np.ndarray]:
        """Dernières features enregistrées pour une tâche."""
        rows = [
            log['features']
            for log in self._logs
            if log.get('task') == task and 'features' in log
        ]
        if not rows:
            return None
        return np.array(rows[-n:], dtype=np.float32)

    def reset(self) -> None:
        """Réinitialise les logs."""
        self._logs = []
