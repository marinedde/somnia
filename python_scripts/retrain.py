#!/usr/bin/env python3
"""
Réentraînement automatisé des modèles Somnia (EEG + ECG).

Prérequis : données prétraitées dans data/processed/ (notebook 02).

Usage :
    python python_scripts/retrain.py
    python python_scripts/retrain.py --task eeg --min-accuracy 0.80
    python python_scripts/retrain.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Racine projet (parent de python_scripts/)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.feature_extractor import FeatureExtractor  # noqa: E402

DATA_EEG = ROOT / "data" / "processed" / "eeg"
DATA_ECG = ROOT / "data" / "processed" / "ecg"
MODELS_DIR = ROOT / "models"
BASELINE_PATH = MODELS_DIR / "baseline_stats.json"
METRICS_PATH = MODELS_DIR / "training_metrics.json"

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def _load_split(data_dir: Path, prefix: str):
    required = [f"X_{prefix}.npy", f"y_{prefix}.npy"]
    for name in required:
        if not (data_dir / name).exists():
            raise FileNotFoundError(f"Fichier manquant : {data_dir / name}")
    return (
        np.load(data_dir / f"X_{prefix}.npy"),
        np.load(data_dir / f"y_{prefix}.npy"),
    )


def _feature_baseline(X: np.ndarray, names: list[str]) -> dict:
    return {
        "n_samples": int(X.shape[0]),
        "mean": np.mean(X, axis=0).tolist(),
        "std": np.std(X, axis=0).tolist(),
        "feature_names": names,
    }


def train_binary(
    task: str,
    data_dir: Path,
    model_out: Path,
    min_auc: float,
) -> dict:
    X_train, y_train = _load_split(data_dir, "train")
    X_val, y_val = _load_split(data_dir, "val")
    X_test, y_test = _load_split(data_dir, "test")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**RF_PARAMS)),
        ]
    )
    pipeline.fit(X_train, y_train)

    def _metrics(X, y):
        y_pred = pipeline.predict(X)
        y_proba = pipeline.predict_proba(X)[:, 1]
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1_apnea": float(f1_score(y, y_pred, pos_label=1)),
            "auc_roc": float(roc_auc_score(y, y_proba)),
        }

    m_val = _metrics(X_val, y_val)
    m_test = _metrics(X_test, y_test)

    if m_val["auc_roc"] < min_auc:
        raise RuntimeError(
            f"[{task}] AUC validation {m_val['auc_roc']:.4f} < seuil {min_auc}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)

    return {
        "task": task,
        "model_path": str(model_out.relative_to(ROOT)),
        "val": m_val,
        "test": m_test,
        "report_test": classification_report(y_test, pipeline.predict(X_test)),
    }


def train_multiclass(
    task: str,
    data_dir: Path,
    model_out: Path,
    min_accuracy: float,
) -> dict:
    X_train, y_train = _load_split(data_dir, "train")
    X_val, y_val = _load_split(data_dir, "val")
    X_test, y_test = _load_split(data_dir, "test")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**RF_PARAMS)),
        ]
    )
    pipeline.fit(X_train, y_train)

    def _metrics(X, y):
        y_pred = pipeline.predict(X)
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1_weighted": float(f1_score(y, y_pred, average="weighted")),
        }

    m_val = _metrics(X_val, y_val)
    m_test = _metrics(X_test, y_test)

    if m_val["accuracy"] < min_accuracy:
        raise RuntimeError(
            f"[{task}] Accuracy validation {m_val['accuracy']:.4f} < seuil {min_accuracy}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)

    return {
        "task": task,
        "model_path": str(model_out.relative_to(ROOT)),
        "val": m_val,
        "test": m_test,
        "report_test": classification_report(y_test, pipeline.predict(X_test)),
    }


def save_artifacts(eeg_result: dict | None, ecg_result: dict | None) -> None:
    baseline = {"updated_at": datetime.now(timezone.utc).isoformat(), "tasks": {}}

    if eeg_result:
        X_train, _ = _load_split(DATA_EEG, "train")
        baseline["tasks"]["sleep_stage"] = _feature_baseline(
            X_train, FeatureExtractor.feature_names()
        )
    if ecg_result:
        X_train, _ = _load_split(DATA_ECG, "train")
        # Features ECG alignées sur 16 dimensions (processed data)
        names = [f"f{i}" for i in range(X_train.shape[1])]
        baseline["tasks"]["apnea"] = _feature_baseline(X_train, names)

    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)

    payload = {
        "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "eeg": eeg_result,
        "ecg": ecg_result,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Baseline stats → {BASELINE_PATH}")
    print(f"Métriques      → {METRICS_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Réentraînement Somnia")
    parser.add_argument(
        "--task",
        choices=["all", "eeg", "ecg"],
        default="all",
        help="Modèle(s) à réentraîner",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.75,
        help="Seuil minimum accuracy validation (EEG)",
    )
    parser.add_argument(
        "--min-auc",
        type=float,
        default=0.65,
        help="Seuil minimum AUC validation (ECG)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Vérifie la présence des données sans entraîner",
    )
    args = parser.parse_args()

    if args.dry_run:
        ready = True
        for d in (DATA_EEG, DATA_ECG):
            if d.exists():
                print(f"[DRY-RUN] OK — {d}")
            else:
                print(f"[DRY-RUN] Manquant : {d}")
                ready = False
        if ready:
            print("[DRY-RUN] Données prêtes pour réentraînement.")
        else:
            print("[DRY-RUN] Script OK — lancez le notebook 02 pour générer les données.")
        return 0

    eeg_result = ecg_result = None

    try:
        if args.task in ("all", "eeg"):
            print("=== Réentraînement EEG (stades de sommeil) ===")
            eeg_result = train_multiclass(
                "eeg",
                DATA_EEG,
                MODELS_DIR / "somnia_eeg_pipeline.joblib",
                args.min_accuracy,
            )
            print(f"Val  : {eeg_result['val']}")
            print(f"Test : {eeg_result['test']}")

        if args.task in ("all", "ecg"):
            print("=== Réentraînement ECG (apnée) ===")
            ecg_result = train_binary(
                "ecg",
                DATA_ECG,
                MODELS_DIR / "somnia_ecg_pipeline.joblib",
                args.min_auc,
            )
            print(f"Val  : {ecg_result['val']}")
            print(f"Test : {ecg_result['test']}")

        save_artifacts(eeg_result, ecg_result)
        print("\n✅ Réentraînement terminé. Redémarrez l'API pour charger les nouveaux modèles.")
        return 0

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Lancez d'abord les notebooks 02 (preprocessing) et 03 (training).")
        return 1
    except RuntimeError as e:
        print(f"❌ Qualité insuffisante — modèle non déployé : {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
