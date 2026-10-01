"""
Tests de l'API Somnia.
Vérifie les endpoints principaux et la validation des inputs.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def eeg_signal_valid():
    """Signal EEG valide — 3000 points."""
    return np.random.randn(3000).tolist()


@pytest.fixture
def ecg_signal_valid():
    """Signal ECG valide — 6000 points."""
    return np.random.randn(6000).tolist()


@pytest.fixture
def eeg_signal_invalid():
    """Signal EEG invalide — mauvaise longueur."""
    return np.random.randn(100).tolist()


@pytest.fixture
def ecg_signal_invalid():
    """Signal ECG invalide — mauvaise longueur."""
    return np.random.randn(100).tolist()


# ─── Mock des modèles pour les tests ──────────────────────────────────────────

@pytest.fixture
def mock_models():
    """Mock des deux modèles ML pour éviter de charger les .joblib."""
    mock_eeg = MagicMock()
    mock_eeg.is_loaded.return_value = True
    mock_eeg.predict.return_value = (
        'N2', 2, 0.89,
        {'Wake':0.05,'N1':0.03,'N2':0.89,'N3':0.02,'REM':0.01},
        'Sommeil léger N2 — Stade le plus fréquent'
    )
    mock_eeg.get_info.return_value = {
        'model_type'   : 'Random Forest',
        'accuracy'     : 0.8344,
        'f1_weighted'  : 0.8302,
        'n_features'   : 16,
        'training_date': '2026-05-07',
        'dataset'      : 'Sleep-EDF',
        'classes'      : ['Wake','N1','N2','N3','REM'],
        'model_loaded' : True,
    }

    mock_ecg = MagicMock()
    mock_ecg.is_loaded.return_value = True
    mock_ecg.predict.return_value = (
        'Normal', 0, 0.92,
        {'Normal':0.92,'Apnée':0.08},
        'Faible', 'Respiration normale détectée avec haute confiance.'
    )
    mock_ecg.get_info.return_value = {
        'model_type'   : 'Random Forest',
        'auc_roc'      : 0.9671,
        'f1_apnea'     : 0.8785,
        'n_features'   : 16,
        'training_date': '2026-05-07',
        'dataset'      : 'Apnea-ECG',
        'classes'      : ['Normal','Apnée'],
        'model_loaded' : True,
    }

    return mock_eeg, mock_ecg


@pytest.fixture
def client(mock_models):
    """Client de test avec modèles mockés."""
    mock_eeg, mock_ecg = mock_models
    with patch('app.main.eeg_model', mock_eeg), \
         patch('app.main.ecg_model', mock_ecg):
        from app.main import app
        with TestClient(app) as c:
            yield c


# ─── Tests endpoints de base ──────────────────────────────────────────────────

class TestRoot:
    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_contains_version(self, client):
        data = client.get("/").json()
        assert "version" in data
        assert data["version"] == "2.0.0"

    def test_root_contains_endpoints(self, client):
        data = client.get("/").json()
        assert "endpoints" in data
        assert "sleep_stage" in data["endpoints"]
        assert "apnea" in data["endpoints"]


class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_structure(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "eeg_model_loaded" in data
        assert "ecg_model_loaded" in data
        assert "version" in data

    def test_health_models_loaded(self, client):
        data = client.get("/health").json()
        assert data["eeg_model_loaded"] is True
        assert data["ecg_model_loaded"] is True
        assert data["status"] == "healthy"


# ─── Tests prédiction EEG ─────────────────────────────────────────────────────

class TestSleepStage:
    def test_predict_valid_signal(self, client, eeg_signal_valid):
        r = client.post("/predict/sleep-stage",
                        json={"signal": eeg_signal_valid})
        assert r.status_code == 200

    def test_predict_returns_class(self, client, eeg_signal_valid):
        data = client.post("/predict/sleep-stage",
                           json={"signal": eeg_signal_valid}).json()
        assert "predicted_class" in data
        assert data["predicted_class"] in ['Wake','N1','N2','N3','REM']

    def test_predict_returns_confidence(self, client, eeg_signal_valid):
        data = client.post("/predict/sleep-stage",
                           json={"signal": eeg_signal_valid}).json()
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_returns_probabilities(self, client, eeg_signal_valid):
        data = client.post("/predict/sleep-stage",
                           json={"signal": eeg_signal_valid}).json()
        assert "probabilities" in data
        probs = data["probabilities"]
        assert set(probs.keys()) == {'Wake','N1','N2','N3','REM'}
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_predict_returns_interpretation(self, client, eeg_signal_valid):
        data = client.post("/predict/sleep-stage",
                           json={"signal": eeg_signal_valid}).json()
        assert "interpretation" in data
        assert len(data["interpretation"]) > 0

    def test_predict_invalid_signal_length(self, client, eeg_signal_invalid):
        r = client.post("/predict/sleep-stage",
                        json={"signal": eeg_signal_invalid})
        assert r.status_code == 422  # Validation error

    def test_predict_empty_signal(self, client):
        r = client.post("/predict/sleep-stage", json={"signal": []})
        assert r.status_code == 422

    def test_predict_missing_signal(self, client):
        r = client.post("/predict/sleep-stage", json={})
        assert r.status_code == 422


# ─── Tests prédiction ECG ─────────────────────────────────────────────────────

class TestApnea:
    def test_predict_valid_signal(self, client, ecg_signal_valid):
        r = client.post("/predict/apnea",
                        json={"signal": ecg_signal_valid})
        assert r.status_code == 200

    def test_predict_returns_class(self, client, ecg_signal_valid):
        data = client.post("/predict/apnea",
                           json={"signal": ecg_signal_valid}).json()
        assert "predicted_class" in data
        assert data["predicted_class"] in ['Normal','Apnée']

    def test_predict_returns_risk_level(self, client, ecg_signal_valid):
        data = client.post("/predict/apnea",
                           json={"signal": ecg_signal_valid}).json()
        assert "risk_level" in data
        assert data["risk_level"] in ['Faible','Modéré','Élevé']

    def test_predict_returns_recommendation(self, client, ecg_signal_valid):
        data = client.post("/predict/apnea",
                           json={"signal": ecg_signal_valid}).json()
        assert "recommendation" in data
        assert len(data["recommendation"]) > 0

    def test_predict_returns_probabilities(self, client, ecg_signal_valid):
        data = client.post("/predict/apnea",
                           json={"signal": ecg_signal_valid}).json()
        assert "probabilities" in data
        probs = data["probabilities"]
        assert set(probs.keys()) == {'Normal','Apnée'}
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_predict_invalid_signal_length(self, client, ecg_signal_invalid):
        r = client.post("/predict/apnea",
                        json={"signal": ecg_signal_invalid})
        assert r.status_code == 422

    def test_predict_empty_signal(self, client):
        r = client.post("/predict/apnea", json={"signal": []})
        assert r.status_code == 422


# ─── Tests monitoring ─────────────────────────────────────────────────────────

class TestClinicalValidation:
    def test_save_validation(self, client, eeg_signal_valid):
        client.post("/predict/sleep-stage", json={"signal": eeg_signal_valid})
        r = client.post("/validations", json={
            "task": "sleep_stage",
            "model_prediction": "N2",
            "model_confidence": 0.89,
            "clinician_verdict": "Confirmé",
            "comment": "Signal propre",
        })
        assert r.status_code == 200
        data = r.json()
        assert "id" in data
        assert data["clinician_verdict"] == "Confirmé"

    def test_list_validations(self, client):
        r = client.get("/validations/recent?n=5")
        assert r.status_code == 200
        assert "validations" in r.json()


class TestFeatureDrift:
    def test_feature_drift_endpoint(self, client, eeg_signal_valid):
        for _ in range(6):
            client.post("/predict/sleep-stage", json={"signal": eeg_signal_valid})
        r = client.get("/monitoring/drift/features?task=sleep_stage")
        assert r.status_code == 200
        assert "drift_detected" in r.json()


class TestMonitoring:
    def test_stats_returns_200(self, client):
        r = client.get("/monitoring/stats")
        assert r.status_code == 200

    def test_stats_structure(self, client):
        data = client.get("/monitoring/stats").json()
        assert "total_predictions" in data
        assert "eeg_predictions" in data
        assert "ecg_predictions" in data

    def test_drift_returns_200(self, client):
        r = client.get("/monitoring/drift")
        assert r.status_code == 200

    def test_drift_structure(self, client):
        data = client.get("/monitoring/drift").json()
        assert "drift_detected" in data
        assert "message" in data

    def test_recent_returns_200(self, client):
        r = client.get("/monitoring/recent")
        assert r.status_code == 200

    def test_reset_returns_200(self, client):
        r = client.delete("/monitoring/reset")
        assert r.status_code == 200


# ─── Tests modèles ML (unitaires) ─────────────────────────────────────────────

class TestFeatureExtractor:
    def test_eeg_features_shape(self):
        from app.feature_extractor import FeatureExtractor
        fe = FeatureExtractor(fs=100, expected_len=3000)
        X  = np.random.randn(5, 3000) * 1e-6
        out = fe.transform(X)
        assert out.shape == (5, 16)

    def test_eeg_features_no_nan(self):
        from app.feature_extractor import FeatureExtractor
        fe  = FeatureExtractor(fs=100, expected_len=3000)
        X   = np.random.randn(3, 3000) * 1e-6
        out = fe.transform(X)
        assert not np.isnan(out).any()

    def test_eeg_features_invalid_shape(self):
        from app.feature_extractor import FeatureExtractor
        fe = FeatureExtractor(fs=100, expected_len=3000)
        with pytest.raises(ValueError):
            fe.transform(np.random.randn(5, 100))


class TestECGFeatureExtractor:
    def test_ecg_features_shape(self):
        from app.ecg_features import ECGFeatureExtractor
        fe  = ECGFeatureExtractor(fs=100, expected_len=6000)
        X   = np.random.randn(5, 6000) * 0.5
        out = fe.transform(X)
        assert out.shape == (5, 16)

    def test_ecg_features_no_nan(self):
        from app.ecg_features import ECGFeatureExtractor
        fe  = ECGFeatureExtractor(fs=100, expected_len=6000)
        X   = np.random.randn(3, 6000) * 0.5
        out = fe.transform(X)
        assert not np.isnan(out).any()

    def test_ecg_features_invalid_shape(self):
        from app.ecg_features import ECGFeatureExtractor
        fe = ECGFeatureExtractor(fs=100, expected_len=6000)
        with pytest.raises(ValueError):
            fe.transform(np.random.randn(5, 100))


class TestMonitoringUnit:
    def test_log_and_stats(self):
        from app.monitoring import SimpleMonitor
        m = SimpleMonitor()
        m.log_prediction('sleep_stage','N2',0.89,{'N2':0.89},15.2)
        m.log_prediction('apnea','Normal',0.92,{'Normal':0.92},12.1)
        stats = m.get_statistics()
        assert stats['total_predictions'] == 2
        assert stats['eeg_predictions']   == 1
        assert stats['ecg_predictions']   == 1

    def test_reset(self):
        from app.monitoring import SimpleMonitor
        m = SimpleMonitor()
        m.log_prediction('sleep_stage','Wake',0.7,{},10.0)
        m.reset()
        stats = m.get_statistics()
        assert stats['total_predictions'] == 0

    def test_drift_insufficient_data(self):
        from app.monitoring import SimpleMonitor
        m      = SimpleMonitor()
        result = m.detect_drift()
        assert result['drift_detected'] is False
        assert 'insuffisant' in result['message'].lower() or 'insuffisantes' in result['message'].lower()
