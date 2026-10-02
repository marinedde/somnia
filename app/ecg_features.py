"""
Feature extractor pour les signaux ECG (détection apnée).

Extrait 16 features d'un segment ECG de 1 minute (6000 points = 60s x 100Hz).
Utilisé dans le pipeline sklearn ECG.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import welch, find_peaks


class ECGFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformeur custom qui extrait les features ECG.

    Features extraites (16 au total) :
    - 5 statistiques signal brut  : mean, std, min, max, range
    - 5 features RR intervals     : mean_RR, std_RR, min_RR, max_RR, RMSSD
    - 3 features HRV fréquentielles: LF_power, HF_power, LF/HF ratio
    - 3 features morphologie QRS  : amplitude_mean, amplitude_std, heart_rate_bpm
    """

    def __init__(self, fs=100, expected_len=6000):
        self.fs           = fs
        self.expected_len = expected_len

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.expected_len:
            raise ValueError(
                f"X doit être (N, {self.expected_len}), reçu {X.shape}"
            )
        features_list = []
        for i, segment in enumerate(X):
            try:
                feats = self._extract(segment)
            except Exception as e:
                print(f"Erreur sample {i}: {e}")
                feats = np.zeros(16)
            features_list.append(feats)
        return np.vstack(features_list)

    def _extract_rr(self, ecg_segment):
        """Détecte les pics R et retourne (intervalles RR en secondes, indices des pics).

        Détecteur : `sleepecg.detect_heartbeats` (robuste, référence du domaine). Le détecteur
        maison (find_peaks sur le signal centré-réduit) s'écartait de plus de 10 bpm de la
        référence sur la moitié des fenêtres SHHS et voyait trois fois trop de variabilité RR
        (0,146 s contre 0,048 s) : les caractéristiques mesuraient le bruit du détecteur.
        Il reste en repli si sleepecg n'est pas installé.
        """
        x = np.asarray(ecg_segment, dtype=np.float64)
        x = x - np.mean(x)
        if abs(x.min()) > abs(x.max()):      # dérivation inversée (fréquent dans SHHS)
            x = -x
        try:
            from sleepecg import detect_heartbeats
            peaks = np.asarray(detect_heartbeats(x, fs=self.fs), dtype=int)
        except ImportError:
            x_norm = x / (np.std(x) + 1e-8)
            peaks, _ = find_peaks(x_norm, height=0.5, distance=int(0.3 * self.fs))
        if len(peaks) < 2:
            return np.array([]), peaks
        rr = np.diff(peaks) / self.fs
        return rr[(rr > 0.3) & (rr < 2.0)], peaks

    def _extract(self, segment):
        """Extrait 16 features d'un segment ECG de 1 minute."""
        features = []

        # === 1. Statistiques signal brut (5 features) ===
        features.append(float(np.mean(segment)))
        features.append(float(np.std(segment)))
        features.append(float(np.min(segment)))
        features.append(float(np.max(segment)))
        features.append(float(np.max(segment) - np.min(segment)))

        # === 2. Features RR intervals (5 features) ===
        rr, peaks = self._extract_rr(segment)
        if len(rr) >= 2:
            features.append(float(np.mean(rr)))
            features.append(float(np.std(rr)))
            features.append(float(np.min(rr)))
            features.append(float(np.max(rr)))
            features.append(float(np.sqrt(np.mean(np.diff(rr) ** 2))))
        else:
            features.extend([0.8, 0.05, 0.6, 1.2, 0.05])

        # === 3. HRV fréquentielle (3 features) ===
        if len(rr) >= 4:
            rr_times   = np.cumsum(rr)
            t_uniform  = np.arange(0, rr_times[-1], 0.25)
            rr_uniform = np.interp(t_uniform, rr_times, rr)
            if len(rr_uniform) > 8:
                freqs_hrv, psd_hrv = welch(
                    rr_uniform, fs=4, nperseg=min(len(rr_uniform), 64)
                )
                lf = float(np.sum(psd_hrv[
                    (freqs_hrv >= 0.04) & (freqs_hrv < 0.15)
                ]))
                hf = float(np.sum(psd_hrv[
                    (freqs_hrv >= 0.15) & (freqs_hrv < 0.40)
                ]))
                features.extend([lf, hf, lf / (hf + 1e-8)])
            else:
                features.extend([0.0, 0.0, 1.0])
        else:
            features.extend([0.0, 0.0, 1.0])

        # === 4. Morphologie QRS (3 features) ===
        if len(peaks) > 0:
            amplitudes = np.abs(segment[peaks] - np.mean(segment))
            features.append(float(np.mean(amplitudes)))
            features.append(float(np.std(amplitudes)))
            features.append(float(
                len(peaks) / (len(segment) / self.fs / 60)
            ))
        else:
            features.extend([0.0, 0.0, 60.0])

        return np.array(features[:16], dtype=np.float32)

    @staticmethod
    def feature_names():
        return [
            'ecg_mean', 'ecg_std', 'ecg_min', 'ecg_max', 'ecg_range',
            'mean_RR', 'std_RR', 'min_RR', 'max_RR', 'RMSSD',
            'LF_power', 'HF_power', 'LF_HF_ratio',
            'qrs_amplitude_mean', 'qrs_amplitude_std', 'heart_rate_bpm',
        ]
