"""
Feature extractor pour les signaux EEG.

Extrait 16 features d'un signal EEG brut (3000 points = 30s x 100Hz).
Utilisé dans le pipeline sklearn — entièrement sérialisable par joblib.

Correction : conversion V → µV avant extraction spectrale.
"""

import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from scipy.signal import welch

log = logging.getLogger(__name__)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformeur custom qui extrait les features EEG.

    Features extraites (16 au total) :
    - 8 statistiques temporelles : mean, std, min, max, Q1, Q3, skewness, kurtosis
    - 5 puissances spectrales    : Delta, Theta, Alpha, Beta, Gamma
    - 3 ratios de puissance      : delta/total, theta/total, alpha/total
    """

    def __init__(self, fs=100, expected_len=3000):
        self.fs           = fs
        self.expected_len = expected_len
        self.n_echecs     = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)      # float16 -> ×1e6 déborderait : float64 d'abord
        if X.ndim != 2 or X.shape[1] != self.expected_len:
            raise ValueError(
                f"X doit être (N, {self.expected_len}), reçu {X.shape}"
            )
        features_list = []
        for i, signal in enumerate(X):
            try:
                feats = self._extract(signal)
            except Exception as e:  # inattendu : on le dit, on compte, zéros reconnaissables
                self.n_echecs += 1
                log.warning("EEG : échec d'extraction sur l'époque %d (%s) -> zéros", i, type(e).__name__)
                feats = np.zeros(16, dtype=np.float32)
            features_list.append(feats)
        out = np.vstack(features_list)
        # Une époque constante donne skewness/kurtosis NaN : remplacés ici, une seule fois,
        # pour l'entraînement comme pour l'API.
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _extract(self, epoch):
        """Extrait 16 features d'une époque EEG de 30s."""
        features = []

        # Conversion V → µV (fix v2)
        epoch_uv = epoch * 1e6

        # === 1. Statistiques temporelles (8 features) ===
        features.append(np.mean(epoch_uv))
        features.append(np.std(epoch_uv))
        features.append(np.min(epoch_uv))
        features.append(np.max(epoch_uv))
        features.append(np.percentile(epoch_uv, 25))
        features.append(np.percentile(epoch_uv, 75))
        features.append(stats.skew(epoch_uv))
        features.append(stats.kurtosis(epoch_uv))

        # === 2. Puissances spectrales (5 features) ===
        freqs, psd = welch(epoch_uv, fs=self.fs, nperseg=256)

        bands = {
            'delta': (0.5, 4),
            'theta': (4,   8),
            'alpha': (8,  13),
            'beta' : (13, 30),
            'gamma': (30, 35),
        }
        band_powers = {}
        for name, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            band_powers[name] = float(np.mean(psd[mask])) if mask.any() else 0.0
            features.append(band_powers[name])

        # === 3. Ratios de puissance (3 features) ===
        total = sum(band_powers.values())
        if total > 0:
            features.append(band_powers['delta'] / total)
            features.append(band_powers['theta'] / total)
            features.append(band_powers['alpha'] / total)
        else:
            features.extend([0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)

    # Noms des features (utile pour SHAP / logs)
    @staticmethod
    def feature_names():
        return [
            'mean', 'std', 'min', 'max', 'q25', 'q75', 'skewness', 'kurtosis',
            'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
            'delta_ratio', 'theta_ratio', 'alpha_ratio',
        ]
